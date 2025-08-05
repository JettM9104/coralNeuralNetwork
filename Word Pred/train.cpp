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
const float LEARNING_RATE = 0.05f;
const float DECAY_RATE = 0.99f;
const float CLIP_VALUE = 1.0f;
const float TEMPERATURE = 1.0f;

float randf() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float dsigmoid(float y) {
    return y * (1 - y);
}

float clip(float val, float min_val, float max_val) {
    return std::max(min_val, std::min(val, max_val));
}

std::vector<float> softmax(const std::vector<float>& logits, float temp = 1.0f) {
    std::vector<float> exps(logits.size());
    float max_logit = *max_element(logits.begin(), logits.end());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp((logits[i] - max_logit) / temp);
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
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

void save_matrix(const std::string& filename, const std::vector<std::vector<float>>& matrix) {
    std::ofstream out(filename);
    for (const auto& row : matrix)
        for (float val : row) out << val << " ";
}

void load_matrix(const std::string& filename, std::vector<std::vector<float>>& matrix) {
    std::ifstream in(filename);
    if (!in) return;
    for (auto& row : matrix)
        for (float& val : row) in >> val;
}

void save_vector(const std::string& filename, const std::vector<float>& vec) {
    std::ofstream out(filename);
    for (float v : vec) out << v << " ";
}

void load_vector(const std::string& filename, std::vector<float>& vec) {
    std::ifstream in(filename);
    if (!in) return;
    for (float& v : vec) in >> v;
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
    words.push_back("<eos>");

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

    std::vector<std::vector<float>> W1(vocab_size, std::vector<float>(EMBED_DIM));
    std::vector<std::vector<float>> W2(EMBED_DIM, std::vector<float>(HIDDEN_DIM));
    std::vector<std::vector<float>> W3(HIDDEN_DIM, std::vector<float>(vocab_size));
    std::vector<float> B2(HIDDEN_DIM, 0);
    std::vector<float> B3(vocab_size, 0);

    std::ifstream check("W1.txt");
    if (check.good()) {
        std::cout << "Loading saved weights...\n";
        load_matrix("W1.txt", W1);
        load_matrix("W2.txt", W2);
        load_matrix("W3.txt", W3);
        load_vector("B2.txt", B2);
        load_vector("B3.txt", B3);
    } else {
        std::cout << "Initializing new weights...\n";
        for (auto& row : W1) for (float& val : row) val = randf();
        for (auto& row : W2) for (float& val : row) val = randf();
        for (auto& row : W3) for (float& val : row) val = randf();
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float loss = 0;
        float lr = LEARNING_RATE * std::pow(DECAY_RATE, epoch / 1000.0f);

        for (size_t i = 0; i < words.size() - 1; ++i) {
            int x = word2idx[words[i]];
            int y = word2idx[words[i + 1]];

            std::vector<float> embed = W1[x];

            std::vector<float> hidden(HIDDEN_DIM, 0);
            for (int j = 0; j < HIDDEN_DIM; ++j)
                for (int k = 0; k < EMBED_DIM; ++k)
                    hidden[j] += embed[k] * W2[k][j];
            for (int j = 0; j < HIDDEN_DIM; ++j)
                hidden[j] = sigmoid(hidden[j] + B2[j]);

            std::vector<float> logits(vocab_size, 0);
            for (int j = 0; j < vocab_size; ++j)
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    logits[j] += hidden[k] * W3[k][j];
            for (int j = 0; j < vocab_size; ++j)
                logits[j] += B3[j];

            std::vector<float> probs = softmax(logits);
            loss -= std::log(probs[y]);

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
                    W3[j][k] -= clip(lr * dlogits[k] * hidden[j], -CLIP_VALUE, CLIP_VALUE);

            for (int j = 0; j < vocab_size; ++j)
                B3[j] -= clip(lr * dlogits[j], -CLIP_VALUE, CLIP_VALUE);

            for (int j = 0; j < EMBED_DIM; ++j)
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    W2[j][k] -= clip(lr * dhidden_act[k] * embed[j], -CLIP_VALUE, CLIP_VALUE);

            for (int j = 0; j < HIDDEN_DIM; ++j)
                B2[j] -= clip(lr * dhidden_act[j], -CLIP_VALUE, CLIP_VALUE);

            for (int j = 0; j < EMBED_DIM; ++j) {
                float grad = 0;
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    grad += dhidden_act[k] * W2[j][k];
                W1[x][j] -= clip(lr * grad, -CLIP_VALUE, CLIP_VALUE);
            }
        }

        if (epoch % 25 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss / words.size() << "\n";
            save_matrix("W1.txt", W1);
            save_matrix("W2.txt", W2);
            save_matrix("W3.txt", W3);
            save_vector("B2.txt", B2);
            save_vector("B3.txt", B3);
        }
    }

    while (true) {
        std::string start_word;
        std::cout << "\nEnter a starting word: ";
        std::cin >> start_word;

        if (word2idx.find(start_word) == word2idx.end()) {
            std::cerr << "Word not found in vocabulary.\n";
            continue;
        }

        int idx = word2idx[start_word];
        std::cout << "Generated text:\n" << start_word << " ";

        for (int i = 0; i < 50; ++i) {
            std::vector<float> embed = W1[idx];
            std::vector<float> hidden(HIDDEN_DIM, 0);
            for (int j = 0; j < HIDDEN_DIM; ++j)
                for (int k = 0; k < EMBED_DIM; ++k)
                    hidden[j] += embed[k] * W2[k][j];
            for (int j = 0; j < HIDDEN_DIM; ++j)
                hidden[j] = sigmoid(hidden[j] + B2[j]);

            std::vector<float> logits(vocab_size, 0);
            for (int j = 0; j < vocab_size; ++j)
                for (int k = 0; k < HIDDEN_DIM; ++k)
                    logits[j] += hidden[k] * W3[k][j];
            for (int j = 0; j < vocab_size; ++j)
                logits[j] += B3[j];

            std::vector<float> probs = softmax(logits, TEMPERATURE);
            idx = sample(probs);

            std::cout << idx2word[idx] << " ";
            if (idx2word[idx] == "<eos>") break;
        }
        std::cout << "\n";
    }

    return 0;
}