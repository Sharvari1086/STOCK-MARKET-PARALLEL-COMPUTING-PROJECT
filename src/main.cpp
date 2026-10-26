
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <omp.h>

using namespace std;

struct StockMetric {
    string name;
    int observations;
    double averageReturn;
    double totalReturn;
    double volatility;
};

vector<string> splitCSV(const string& line) {
    vector<string> fields;
    string field;
    stringstream ss(line);

    while (getline(ss, field, ',')) {
        fields.push_back(field);
    }

    if (!line.empty() && line.back() == ',') {
        fields.push_back("");
    }

    return fields;
}

bool parsePrice(const string& value, double& price) {
    try {
        size_t used = 0;
        price = stod(value, &used);

        return used == value.size() &&
               isfinite(price) &&
               price > 0.0;
    }
    catch (...) {
        return false;
    }
}

StockMetric calculateMetrics(
    const string& name,
    const vector<double>& prices
) {
    StockMetric metric;
    metric.name = name;
    metric.observations = static_cast<int>(prices.size()) - 1;
    metric.averageReturn = 0.0;
    metric.totalReturn = 0.0;
    metric.volatility = 0.0;

    vector<double> returns;
    returns.reserve(prices.size() - 1);

    for (size_t i = 1; i < prices.size(); i++) {
        double dailyReturn = prices[i] / prices[i - 1] - 1.0;
        returns.push_back(dailyReturn);
    }

    double averageReturn = 0.0;

    for (double r : returns) {
        averageReturn += r;
    }

    averageReturn /= returns.size();

    double variance = 0.0;

    for (double r : returns) {
        double difference = r - averageReturn;
        variance += difference * difference;
    }

    variance /= returns.size();

    metric.averageReturn = averageReturn * 100.0;
    metric.totalReturn =
        (prices.back() / prices.front() - 1.0) * 100.0;
    metric.volatility = sqrt(variance) * 100.0;

    return metric;
}

vector<StockMetric> sequentialMetrics(
    const vector<string>& names,
    const vector<vector<double>>& stockPrices
) {
    vector<StockMetric> metrics(names.size());

    for (int i = 0; i < static_cast<int>(names.size()); i++) {
        metrics[i] = calculateMetrics(names[i], stockPrices[i]);
    }

    return metrics;
}

vector<StockMetric> parallelMetrics(
    const vector<string>& names,
    const vector<vector<double>>& stockPrices,
    int threads
) {
    vector<StockMetric> metrics(names.size());

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
        metrics[i] = calculateMetrics(names[i], stockPrices[i]);
    }

    return metrics;
}

int main() {
    filesystem::create_directories("results");

    ifstream file("data/stock_prices.csv");

    if (!file.is_open()) {
        cerr << "Error: Could not open data/stock_prices.csv\n";
        return 1;
    }

    string line;

    if (!getline(file, line)) {
        cerr << "Error: CSV file is empty\n";
        return 1;
    }

    vector<string> header = splitCSV(line);
    vector<vector<double>> prices;

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        vector<string> row = splitCSV(line);

        if (row.size() != header.size()) {
            continue;
        }

        vector<double> values;
        values.reserve(header.size() - 1);

        for (size_t j = 1; j < row.size(); j++) {
            double price = 0.0;

            if (!parsePrice(row[j], price)) {
                price = NAN;
            }

            values.push_back(price);
        }

        prices.push_back(values);
    }

    file.close();

    if (prices.size() < 2) {
        cerr << "Error: Not enough price observations\n";
        return 1;
    }

    int totalDates = static_cast<int>(prices.size());
    int totalStocks = static_cast<int>(header.size()) - 1;

    vector<string> stockNames;
    vector<vector<double>> stockPrices;

    for (int j = 0; j < totalStocks; j++) {
        vector<double> series;
        bool valid = true;

        for (const auto& row : prices) {
            double price = row[j];

            if (!isfinite(price) || price <= 0.0) {
                valid = false;
                break;
            }

            series.push_back(price);
        }

        if (valid && series.size() >= 2) {
            stockNames.push_back(header[j + 1]);
            stockPrices.push_back(series);
        }
    }

    int validStocks = static_cast<int>(stockNames.size());

    if (validStocks == 0) {
        cerr << "Error: No stocks have complete valid price histories\n";
        return 1;
    }

    const int repetitions = 3;
    const int threads = max(1, omp_get_max_threads());

    double sequentialTotal = 0.0;
    double parallelTotal = 0.0;

    vector<StockMetric> sequentialResult;
    vector<StockMetric> parallelResult;

    for (int r = 0; r < repetitions; r++) {
        double start = omp_get_wtime();

        sequentialResult = sequentialMetrics(stockNames, stockPrices);

        sequentialTotal += omp_get_wtime() - start;

        start = omp_get_wtime();

        parallelResult = parallelMetrics(
            stockNames, stockPrices, threads
        );

        parallelTotal += omp_get_wtime() - start;
    }

    double sequentialTime = sequentialTotal / repetitions;
    double parallelTime = parallelTotal / repetitions;

    double speedup = sequentialTime / parallelTime;
    double efficiency = (speedup / threads) * 100.0;

    double maxDifference = 0.0;

    for (int i = 0; i < validStocks; i++) {
        maxDifference = max(
            maxDifference,
            abs(sequentialResult[i].averageReturn -
                parallelResult[i].averageReturn)
        );

        maxDifference = max(
            maxDifference,
            abs(sequentialResult[i].totalReturn -
                parallelResult[i].totalReturn)
        );

        maxDifference = max(
            maxDifference,
            abs(sequentialResult[i].volatility -
                parallelResult[i].volatility)
        );
    }

    ofstream output("results/stock_metrics.csv");

    if (!output.is_open()) {
        cerr << "Error: Could not create stock_metrics.csv\n";
        return 1;
    }

    output << "Stock,Observations,AverageDailyReturnPercent,"
           << "TotalReturnPercent,VolatilityPercent\n";

    output << fixed << setprecision(8);

    for (const auto& metric : parallelResult) {
        output << metric.name << ","
               << metric.observations << ","
               << metric.averageReturn << ","
               << metric.totalReturn << ","
               << metric.volatility << "\n";
    }

    output.close();

    ofstream performance("results/stock_metrics_performance.csv");

    if (!performance.is_open()) {
        cerr << "Error: Could not create stock_metrics_performance.csv\n";
        return 1;
    }

    performance
        << "SequentialTime,ParallelTime,Threads,Repetitions,"
        << "Speedup,EfficiencyPercent,MaxDifference\n";

    performance << setprecision(10)
                << sequentialTime << ","
                << parallelTime << ","
                << threads << ","
                << repetitions << ","
                << speedup << ","
                << efficiency << ","
                << maxDifference << "\n";

    performance.close();

    cout << fixed << setprecision(6);

    cout << "\nSTOCK MARKET METRICS\n";
    cout << "Total price observations: " << totalDates << "\n";
    cout << "Total stock columns: " << totalStocks << "\n";
    cout << "Valid stocks processed: " << validStocks << "\n";

    cout << "\nSEQUENTIAL VS PARALLEL PERFORMANCE\n";
    cout << "Sequential time: " << sequentialTime << " seconds\n";
    cout << "Parallel time: " << parallelTime << " seconds\n";
    cout << "OpenMP threads: " << threads << "\n";
    cout << "Speedup: " << speedup << "x\n";
    cout << "Parallel efficiency: " << efficiency << "%\n";
    cout << "Maximum result difference: " << maxDifference << "\n";

    cout << "\nFirst 10 stock results:\n";

    for (int i = 0; i < min(10, validStocks); i++) {
        const auto& metric = parallelResult[i];

        cout << metric.name
             << " | Average daily return: "
             << metric.averageReturn << "%"
             << " | Total return: "
             << metric.totalReturn << "%"
             << " | Volatility: "
             << metric.volatility << "%\n";
    }

    cout << "\nSaved results/stock_metrics.csv\n";
    cout << "Saved results/stock_metrics_performance.csv\n";

    return 0;
}
