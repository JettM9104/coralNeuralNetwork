#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include <ctime>
#include <algorithm>

// Hyperparameters
const int EMBED_DIM = 32;
const int HIDDEN_DIM = 64;
const int EPOCHS = 0;
const float LEARNING_RATE = 0.05;

// Utility: Random float in [-1, 1]
float randf() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

// Activation functions
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
float dsigmoid(float y) {
    return y * (1 - y);
}

// Softmax
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exps(logits.size());
    float max_logit = *max_element(logits.begin(), logits.end());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (float& v : exps) v /= sum;
    return exps;
}

// Sample from probability distribution
int sample(const std::vector<float>& probs) {
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return probs.size() - 1;
}

// Tokenize input text into words
std::vector<std::string> tokenize(const std::string& text) {
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

// Save and load functions
void save_matrix(const std::string& filename, const std::vector<std::vector<float>>& matrix) {
    std::ofstream out(filename);
    for (const auto& row : matrix) {
        for (float val : row) {
            out << val << " ";
        }
        out << "\n";
    }
}

void load_matrix(const std::string& filename, std::vector<std::vector<float>>& matrix) {
    std::ifstream in(filename);
    if (!in) return;
    for (auto& row : matrix) {
        for (float& val : row) {
            in >> val;
        }
    }
}

int main() {
    srand(time(0));

    // Load text
    std::ifstream file("text.txt");
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        std::cerr << "text.txt is empty or missing.\n";
        return 1;
    }

    std::vector<std::string> words = tokenize(text);

    // Build vocabulary
    std::map<std::string, int> word2idx;
    std::map<int, std::string> idx2word;
    for (const std::string& w : words) word2idx[w] = 0;
    int id = 0;
    for (auto& [w, _] : word2idx) {
        word2idx[w] = id;
        idx2word[id] = w;
        id++;
    }
    int vocab_size = word2idx.size();

    // Initialize weights
    std::vector<std::vector<float>> W1(vocab_size, std::vector<float>(EMBED_DIM));
    std::vector<std::vector<float>> W2(EMBED_DIM, std::vector<float>(HIDDEN_DIM));
    std::vector<std::vector<float>> W3(HIDDEN_DIM, std::vector<float>(vocab_size));

    // Try loading saved weights
    std::ifstream check("W1.txt");
    if (check.good()) {
        std::cout << "Loading saved weights...\n";
        load_matrix("W1.txt", W1);
        load_matrix("W2.txt", W2);
        load_matrix("W3.txt", W3);
    } else {
        std::cout << "Initializing new weights...\n";
        for (auto& row : W1) for (float& val : row) val = randf();
        for (auto& row : W2) for (float& val : row) val = randf();
        for (auto& row : W3) for (float& val : row) val = randf();
    }

    // Training
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float loss = 0;

        for (size_t i = 0; i < words.size() - 1; ++i) {
            int x = word2idx[words[i]];
            int y = word2idx[words[i + 1]];

            // Forward
            std::vector<float> embed = W1[x];

            std::vector<float> hidden(HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                hidden[j] = 0;
                for (int k = 0; k < EMBED_DIM; ++k)
                    hidden[j] += embed[k] * W2[k][j];
                hidden[j] = sigmoid(hidden[j]);
            }

            std::vector<float> logits(vocab_size);
            for (int j = 0; j < vocab_size; ++j) {
                logits[j] = 0;
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    logits[j] += hidden[k] * W3[k][j];
            }

            std::vector<float> probs = softmax(logits);
            loss -= std::log(probs[y]);

            // Backward
            std::vector<float> dlogits = probs;
            dlogits[y] -= 1;

            std::vector<float> dhidden(HIDDEN_DIM, 0);
            for (int j = 0; j < HIDDEN_DIM; ++j)
                for (int k = 0; k < vocab_size; ++k)
                    dhidden[j] += dlogits[k] * W3[j][k];

            std::vector<float> dhidden_act(HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM; ++j)
                dhidden_act[j] = dhidden[j] * dsigmoid(hidden[j]);

            for (int j = 0; j < HIDDEN_DIM; ++j)
                for (int k = 0; k < vocab_size; ++k)
                    W3[j][k] -= LEARNING_RATE * dlogits[k] * hidden[j];

            for (int j = 0; j < EMBED_DIM; ++j)
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    W2[j][k] -= LEARNING_RATE * dhidden_act[k] * embed[j];

            for (int j = 0; j < EMBED_DIM; ++j) {
                float grad = 0;
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    grad += dhidden_act[k] * W2[j][k];
                W1[x][j] -= LEARNING_RATE * grad;
            }
        }

        std::cout << "Epoch " << epoch << ", Loss: " << loss / words.size() << "\n";
        if (epoch % 10 == 0) {
            // Save weights
            save_matrix("W1.txt", W1);
            save_matrix("W2.txt", W2);
            save_matrix("W3.txt", W3);
            std::cout << "Weights saved to W1.txt, W2.txt, W3.txt\n";
        } 
    }

    // Save weights
    save_matrix("W1.txt", W1);
    save_matrix("W2.txt", W2);
    save_matrix("W3.txt", W3);
    std::cout << "Weights saved to W1.txt, W2.txt, W3.txt\n";
    while (true) {
        // Interactive generation
        std::string start_word;
        std::cout << "\nEnter a starting word: ";
        std::cin >> start_word;

        if (word2idx.find(start_word) == word2idx.end()) {
            std::cerr << "Word not found in vocabulary.\n";
            return 1;
        }

        int idx = word2idx[start_word];

        std::cout << "Generated text:\n" << start_word << " ";

        for (int i = 0; i < 50; ++i) {
            std::vector<float> embed = W1[idx];

            std::vector<float> hidden(HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                hidden[j] = 0;
                for (int k = 0; k < EMBED_DIM; ++k)
                    hidden[j] += embed[k] * W2[k][j];
                hidden[j] = sigmoid(hidden[j]);
            }

            std::vector<float> logits(vocab_size);
            for (int j = 0; j < vocab_size; ++j) {
                logits[j] = 0;
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    logits[j] += hidden[k] * W3[k][j];
            }

            std::vector<float> probs = softmax(logits);
            idx = sample(probs);

            std::cout << idx2word[idx] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}
