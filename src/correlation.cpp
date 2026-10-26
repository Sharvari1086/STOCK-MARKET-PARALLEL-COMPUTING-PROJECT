
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <omp.h>

using namespace std;

vector<string> splitCSV(const string& line) {
    vector<string> values;
    string value;
    stringstream ss(line);

    while (getline(ss, value, ',')) {
        values.push_back(value);
    }

    return values;
}

double parsePrice(const string& value) {
    if (value.empty() || value == "null" || value == "NA") {
        return numeric_limits<double>::quiet_NaN();
    }

    try {
        size_t pos;
        double price = stod(value, &pos);

        if (pos != value.size() || price <= 0.0) {
            return numeric_limits<double>::quiet_NaN();
        }

        return price;
    } catch (...) {
        return numeric_limits<double>::quiet_NaN();
    }
}

vector<double> sequentialCorrelation(
    const vector<double>& normalizedReturns,
    int days,
    int n
) {
    vector<double> matrix(n * n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;

            for (int t = 0; t < days; t++) {
                sum += normalizedReturns[i * days + t] *
                       normalizedReturns[j * days + t];
            }

            matrix[i * n + j] = sum;
            matrix[j * n + i] = sum;
        }
    }

    return matrix;
}

vector<double> parallelCorrelation(
    const vector<double>& normalizedReturns,
    int days,
    int n,
    int threads,
    int blockSize
) {
    vector<double> matrix(n * n, 0.0);
    int numBlocks = (n + blockSize - 1) / blockSize;

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int bi = 0; bi < numBlocks; bi++) {
        for (int bj = bi; bj < numBlocks; bj++) {
            int rowStart = bi * blockSize;
            int rowEnd = min(rowStart + blockSize, n);
            int colStart = bj * blockSize;
            int colEnd = min(colStart + blockSize, n);

            for (int i = rowStart; i < rowEnd; i++) {
                int jStart = (bi == bj) ? max(i, colStart) : colStart;

                for (int j = jStart; j < colEnd; j++) {
                    double sum = 0.0;

                    for (int t = 0; t < days; t++) {
                        sum += normalizedReturns[i * days + t] *
                               normalizedReturns[j * days + t];
                    }

                    matrix[i * n + j] = sum;
                    matrix[j * n + i] = sum;
                }
            }
        }
    }

    return matrix;
}

int main() {
    filesystem::create_directories("results");

    ifstream input("data/stock_prices.csv");

    if (!input.is_open()) {
        cerr << "Error: Cannot open data/stock_prices.csv\n";
        return 1;
    }

    string line;

    if (!getline(input, line)) {
        cerr << "Error: CSV file is empty.\n";
        return 1;
    }

    vector<string> headers = splitCSV(line);
    vector<vector<double>> prices;
    vector<string> dates;

    while (getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        vector<string> values = splitCSV(line);

        if (values.size() != headers.size()) {
            continue;
        }

        dates.push_back(values[0]);

        vector<double> row;

        for (size_t j = 1; j < values.size(); j++) {
            row.push_back(parsePrice(values[j]));
        }

        prices.push_back(row);
    }

    input.close();

    if (prices.size() < 2) {
        cerr << "Error: Not enough price observations.\n";
        return 1;
    }

    int totalDays = static_cast<int>(prices.size());
    int totalStocks = static_cast<int>(headers.size()) - 1;

    vector<string> stockNames;
    vector<vector<double>> stockPrices;

    for (int j = 0; j < totalStocks; j++) {
        vector<double> series;
        bool valid = true;

        for (int t = 0; t < totalDays; t++) {
            double price = prices[t][j];

            if (!isfinite(price) || price <= 0.0) {
                valid = false;
                break;
            }

            series.push_back(price);
        }

        if (valid) {
            stockNames.push_back(headers[j + 1]);
            stockPrices.push_back(series);
        }
    }

    int n = static_cast<int>(stockPrices.size());

    if (n < 2) {
        cerr << "Error: Not enough stocks with complete price histories.\n";
        return 1;
    }

    int days = totalDays - 1;

    vector<double> normalizedReturns(n * days, 0.0);
    vector<string> eligibleNames;
    vector<int> eligibleIndices;

    for (int i = 0; i < n; i++) {
        vector<double> returns(days);
        double mean = 0.0;

        for (int t = 0; t < days; t++) {
            returns[t] =
                (stockPrices[i][t + 1] - stockPrices[i][t]) /
                stockPrices[i][t];

            mean += returns[t];
        }

        mean /= days;

        double variance = 0.0;

        for (int t = 0; t < days; t++) {
            double difference = returns[t] - mean;
            variance += difference * difference;
        }

        variance /= days;
        double sd = sqrt(variance);

        if (sd <= 1e-12) {
            continue;
        }

        eligibleNames.push_back(stockNames[i]);
        eligibleIndices.push_back(i);

        int newIndex = static_cast<int>(eligibleNames.size()) - 1;

        for (int t = 0; t < days; t++) {
            normalizedReturns[newIndex * days + t] =
                (returns[t] - mean) / (sd * sqrt(static_cast<double>(days)));
        }
    }

    stockNames = eligibleNames;
    n = static_cast<int>(stockNames.size());

    normalizedReturns.resize(n * days);

    if (n < 2) {
        cerr << "Error: Not enough stocks with varying returns.\n";
        return 1;
    }

    cout << fixed << setprecision(6);
    cout << "\nSTOCK CORRELATION ANALYSIS\n";
    cout << "Eligible stocks: " << n << "\n";
    cout << "Daily return observations: " << days << "\n";
    cout << "Available OpenMP threads: " << omp_get_max_threads() << "\n";

    const int blockSize = 32;
    const int repetitions = 3;

    double sequentialTotal = 0.0;
    vector<double> sequentialResult;

    for (int r = 0; r < repetitions; r++) {
        double start = omp_get_wtime();

        sequentialResult = sequentialCorrelation(
            normalizedReturns, days, n
        );

        sequentialTotal += omp_get_wtime() - start;
    }

    double sequentialTime = sequentialTotal / repetitions;

    int threadCounts[] = {1, 2, 4, 8, 16};

    ofstream scalabilityFile("results/scalability.csv");

    if (!scalabilityFile.is_open()) {
        cerr << "Error: Cannot create results/scalability.csv\n";
        return 1;
    }

    scalabilityFile
        << "Threads,ParallelTime,Speedup,EfficiencyPercent\n";

    vector<double> finalParallelResult;
    double bestParallelTime = numeric_limits<double>::max();
    int bestThreads = 1;

    cout << "\nSEQUENTIAL TIME: " << sequentialTime << " seconds\n";
    cout << "\nSCALABILITY RESULTS\n";

    for (int threads : threadCounts) {
        double parallelTotal = 0.0;
        vector<double> parallelResult;

        for (int r = 0; r < repetitions; r++) {
            double start = omp_get_wtime();

            parallelResult = parallelCorrelation(
                normalizedReturns, days, n, threads, blockSize
            );

            parallelTotal += omp_get_wtime() - start;
        }

        double parallelTime = parallelTotal / repetitions;
        double speedup = sequentialTime / parallelTime;
        double efficiency = speedup / threads * 100.0;

        scalabilityFile
            << threads << ","
            << parallelTime << ","
            << speedup << ","
            << efficiency << "\n";

        cout << "Threads: " << threads
             << " | Parallel time: " << parallelTime << " s"
             << " | Speedup: " << speedup << "x"
             << " | Efficiency: " << efficiency << "%\n";

        if (threads == 1) {
            finalParallelResult = parallelResult;
        }

        if (parallelTime < bestParallelTime) {
            bestParallelTime = parallelTime;
            bestThreads = threads;
        }
    }

    scalabilityFile.close();

    double maxDifference = 0.0;
    double maxDiagonalError = 0.0;
    double minCorrelation = 1.0;
    double maxCorrelation = -1.0;

    vector<double> correlationMatrix = sequentialResult;

    for (size_t i = 0; i < sequentialResult.size(); i++) {
        maxDifference = max(
            maxDifference,
            abs(sequentialResult[i] - finalParallelResult[i])
        );
    }

    for (int i = 0; i < n; i++) {
        maxDiagonalError = max(
            maxDiagonalError,
            abs(sequentialResult[i * n + i] - 1.0)
        );

        for (int j = 0; j < n; j++) {
            double value = sequentialResult[i * n + j];
            minCorrelation = min(minCorrelation, value);
            maxCorrelation = max(maxCorrelation, value);
        }
    }

    ofstream matrixFile("results/correlation_matrix.csv");

    if (!matrixFile.is_open()) {
        cerr << "Error: Cannot create correlation_matrix.csv\n";
        return 1;
    }

    matrixFile << "Stock";

    for (const string& name : stockNames) {
        matrixFile << "," << name;
    }

    matrixFile << "\n";

    for (int i = 0; i < n; i++) {
        matrixFile << stockNames[i];

        for (int j = 0; j < n; j++) {
            matrixFile << "," << correlationMatrix[i * n + j];
        }

        matrixFile << "\n";
    }

    matrixFile.close();

    ofstream performanceFile("results/performance.csv");

    performanceFile
        << "SequentialTime,ParallelTime,Threads,BlockSize,Repetitions,"
        << "Speedup,EfficiencyPercent,MaxDifference,MaxDiagonalError,"
        << "MinCorrelation,MaxCorrelation\n";

    double bestSpeedup = sequentialTime / bestParallelTime;
    double bestEfficiency = bestSpeedup / bestThreads * 100.0;

    performanceFile
        << sequentialTime << ","
        << bestParallelTime << ","
        << bestThreads << ","
        << blockSize << ","
        << repetitions << ","
        << bestSpeedup << ","
        << bestEfficiency << ","
        << maxDifference << ","
        << maxDiagonalError << ","
        << minCorrelation << ","
        << maxCorrelation << "\n";

    performanceFile.close();

    cout << "\nCORRECTNESS CHECK\n";
    cout << "Maximum difference: " << maxDifference << "\n";
    cout << "Maximum diagonal error: " << maxDiagonalError << "\n";
    cout << "Correlation range: [" << minCorrelation
         << ", " << maxCorrelation << "]\n";

    cout << "\nBest measured parallel time: " << bestParallelTime
         << " seconds using " << bestThreads << " threads.\n";

    cout << "\nSaved results/correlation_matrix.csv\n";
    cout << "Saved results/performance.csv\n";
    cout << "Saved results/scalability.csv\n";

    return 0;
}
