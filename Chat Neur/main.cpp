// NOTE: This is a condensed final version of your code including the generation fix.

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

const int EMBED_DIM = 32;
const int HIDDEN_DIM = 64;
const int EPOCHS = 1000000;
const int BATCH_SIZE = 8;
const float LEARNING_RATE = 0.05f;
const int MAX_ANSWER_LEN = 30;

float randf() { return ((float)rand() / RAND_MAX) * 2 - 1; }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float dsigmoid(float y) { return y * (1 - y); }

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exps(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (float& v : exps) v /= sum;
    return exps;
}

int sample(const std::vector<float>& probs) {
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return probs.size() - 1;
}

std::vector<std::string> tokenize(const std::string& text) {
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) words.push_back(word);
    return words;
}

void save_matrix(const std::string& filename, const std::vector<std::vector<float>>& matrix) {
    std::ofstream out(filename);
    for (const auto& row : matrix) {
        for (float val : row) out << val << " ";
        out << "\n";
    }
}

void load_matrix(const std::string& filename, std::vector<std::vector<float>>& matrix) {
    std::ifstream in(filename);
    if (!in) return;
    for (auto& row : matrix)
        for (float& val : row) in >> val;
}

float cross_entropy_loss(const std::vector<float>& probs, int target_idx) {
    float p = std::max(probs[target_idx], 1e-10f);
    return -std::log(p);
}

float update_weights_batch(
    std::vector<std::vector<float>>& W1,
    std::vector<std::vector<float>>& W2,
    std::vector<std::vector<float>>& W3,
    const std::vector<int>& batch_x,
    const std::vector<int>& batch_y,
    int vocab_size
) {
    std::vector<std::vector<float>> dW1(W1.size(), std::vector<float>(EMBED_DIM, 0));
    std::vector<std::vector<float>> dW2(EMBED_DIM, std::vector<float>(HIDDEN_DIM, 0));
    std::vector<std::vector<float>> dW3(HIDDEN_DIM, std::vector<float>(vocab_size, 0));

    float total_loss = 0.0f;

    for (size_t sample_idx = 0; sample_idx < batch_x.size(); ++sample_idx) {
        int x = batch_x[sample_idx];
        int y = batch_y[sample_idx];
        const auto& embed = W1[x];

        std::vector<float> hidden(HIDDEN_DIM, 0);
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            for (int k = 0; k < EMBED_DIM; ++k)
                hidden[j] += embed[k] * W2[k][j];
            hidden[j] = sigmoid(hidden[j]);
        }

        std::vector<float> logits(vocab_size, 0);
        for (int j = 0; j < vocab_size; ++j)
            for (int k = 0; k < HIDDEN_DIM; ++k)
                logits[j] += hidden[k] * W3[k][j];

        auto probs = softmax(logits);
        total_loss += cross_entropy_loss(probs, y);

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
                dW3[j][k] += dlogits[k] * hidden[j];

        for (int j = 0; j < EMBED_DIM; ++j)
            for (int k = 0; k < HIDDEN_DIM; ++k)
                dW2[j][k] += dhidden_act[k] * embed[j];

        for (int j = 0; j < EMBED_DIM; ++j) {
            float grad = 0;
            for (int k = 0; k < HIDDEN_DIM; ++k)
                grad += dhidden_act[k] * W2[j][k];
            dW1[x][j] += grad;
        }
    }

    float inv_batch = 1.0f / batch_x.size();
    for (size_t i = 0; i < W3.size(); ++i)
        for (size_t j = 0; j < W3[0].size(); ++j)
            W3[i][j] -= LEARNING_RATE * dW3[i][j] * inv_batch;
    for (size_t i = 0; i < W2.size(); ++i)
        for (size_t j = 0; j < W2[0].size(); ++j)
            W2[i][j] -= LEARNING_RATE * dW2[i][j] * inv_batch;
    for (size_t i = 0; i < W1.size(); ++i)
        for (size_t j = 0; j < W1[0].size(); ++j)
            W1[i][j] -= LEARNING_RATE * dW1[i][j] * inv_batch;

    return total_loss / batch_x.size();
}

std::string generate(const std::string& prompt, const std::map<std::string, int>& word2idx,
                     const std::map<int, std::string>& idx2word,
                     const std::vector<std::vector<float>>& W1,
                     const std::vector<std::vector<float>>& W2,
                     const std::vector<std::vector<float>>& W3,
                     int max_len = 20) {
    std::vector<std::string> input_words = tokenize(prompt);
    std::string last_word = input_words.back();
    if (!word2idx.count(last_word)) return "[Unknown token]";

    int token = word2idx.at(last_word);
    std::string output;

    for (int step = 0; step < max_len; ++step) {
        const auto& embed = W1[token];
        std::vector<float> hidden(HIDDEN_DIM, 0);
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            for (int k = 0; k < EMBED_DIM; ++k)
                hidden[j] += embed[k] * W2[k][j];
            hidden[j] = sigmoid(hidden[j]);
        }

        std::vector<float> logits(W3[0].size(), 0);
        for (int j = 0; j < logits.size(); ++j)
            for (int k = 0; k < HIDDEN_DIM; ++k)
                logits[j] += hidden[k] * W3[k][j];

        if (step == 0) logits[word2idx.at("<eos>")] = -1e9;

        auto probs = softmax(logits);
        token = sample(probs);
        std::string word = idx2word.at(token);
        if (word == "<eos>") break;
        output += word + " ";
    }
    return output;
}
