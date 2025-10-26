#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// Gini Impurity
double gini_impurity(const vector<int>& labels) {
    if (labels.empty()) return 0.0;
    unordered_map<int, int> freq;
    for (int l : labels) freq[l]++;
    double impurity = 1.0;
    for (auto& p : freq) {
        double prob = (double)p.second / labels.size();
        impurity -= prob * prob;
    }
    return impurity;
}

// Decision Tree
struct DecisionTree {
    int feature_index;
    double threshold;
    int left_label, right_label;
    bool is_leaf;

    DecisionTree() : is_leaf(true), left_label(-1), right_label(-1) {}

    int majority_label(const vector<int>& y) {
        unordered_map<int, int> count;
        for (int l : y) count[l]++;
        int best_label = -1, best_count = -1;
        for (auto& c : count)
            if (c.second > best_count)
                best_label = c.first, best_count = c.second;
        return best_label;
    }

    void train(const vector<vector<double>>& X, const vector<int>& y) {
        int n = X.size(), m = X[0].size();
        double best_gini = 1e9;
        int best_feat = 0;
        double best_thresh = 0;

        for (int f = 0; f < m; f++) {
            vector<double> vals;
            for (int i = 0; i < n; i++) vals.push_back(X[i][f]);
            sort(vals.begin(), vals.end());
            for (int t = 1; t < n; t++) {
                double thresh = (vals[t] + vals[t - 1]) / 2.0;
                vector<int> left_labels, right_labels;
                for (int i = 0; i < n; i++) {
                    if (X[i][f] <= thresh) left_labels.push_back(y[i]);
                    else right_labels.push_back(y[i]);
                }
                double gini = (left_labels.size() * gini_impurity(left_labels) +
                               right_labels.size() * gini_impurity(right_labels)) / n;
                if (gini < best_gini) {
                    best_gini = gini;
                    best_feat = f;
                    best_thresh = thresh;
                }
            }
        }

        feature_index = best_feat;
        threshold = best_thresh;
        is_leaf = false;

        vector<int> left_labels, right_labels;
        for (int i = 0; i < n; i++) {
            if (X[i][feature_index] <= threshold)
                left_labels.push_back(y[i]);
            else
                right_labels.push_back(y[i]);
        }

        left_label = majority_label(left_labels);
        right_label = majority_label(right_labels);
    }

    int predict(const vector<double>& x) const {
        if (is_leaf) return left_label;
        return x[feature_index] <= threshold ? left_label : right_label;
    }
};

// Parallel Random Forest
struct RandomForestParallel {
    int n_trees;
    vector<DecisionTree> trees;

    RandomForestParallel(int n) : n_trees(n) {}

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        int n = X.size();
        trees.resize(n_trees);

        #pragma omp parallel for
        for (int t = 0; t < n_trees; t++) {
            vector<vector<double>> Xs;
            vector<int> ys;

            unsigned seed = omp_get_thread_num() + 100;
            srand(seed);

            for (int i = 0; i < n; i++) {
                int idx = rand() % n;
                Xs.push_back(X[idx]);
                ys.push_back(y[idx]);
            }

            DecisionTree local_tree;
            local_tree.train(Xs, ys);
            trees[t] = local_tree;
        }
    }

    int predict(const vector<double>& x) {
        unordered_map<int, int> votes;
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int t = 0; t < n_trees; t++) {
                int label = trees[t].predict(x);
                #pragma omp critical
                votes[label]++;
            }
        }
        int best_label = -1, best_count = -1;
        for (auto& v : votes)
            if (v.second > best_count)
                best_label = v.first, best_count = v.second;
        return best_label;
    }
};

int main() {
    srand(42);
    int n_samples, n_features;
    cout << "Enter number of samples: ";
    cin >> n_samples;
    cout << "Enter number of features: ";
    cin >> n_features;

    vector<vector<double>> X(n_samples, vector<double>(n_features));
    vector<int> y(n_samples);

    cout << "\nEnter training data (features followed by label):\n";
    for (int i = 0; i < n_samples; i++) {
        cout << "Sample " << i + 1 << ": ";
        for (int j = 0; j < n_features; j++) cin >> X[i][j];
        cin >> y[i];
    }

    int n_trees;
    cout << "\nEnter number of trees in Random Forest: ";
    cin >> n_trees;

    RandomForestParallel rf(n_trees);
    double start = omp_get_wtime();
    rf.fit(X, y);
    double end = omp_get_wtime();

    int n_tests;
    cout << "\nEnter number of test samples: ";
    cin >> n_tests;

    for (int i = 0; i < n_tests; i++) {
        vector<double> test(n_features);
        cout << "Enter test sample " << i + 1 << ": ";
        for (int j = 0; j < n_features; j++) cin >> test[j];
        int pred = rf.predict(test);
        cout << "Predicted class: " << pred << endl;
    }

    cout << "\nTraining time: " << (end - start) << " seconds\n";
    return 0;
}
