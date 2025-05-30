#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <ctime>

// Simple neural net parameters
const int EMBED_DIM = 32;
const int HIDDEN_DIM = 64;
const int EPOCHS = 20;
const float LEARNING_RATE = 0.05;

// Random float in [-1, 1]
float randf() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

// Activation
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
float dsigmoid(float y) {
    return y * (1 - y); // derivative assuming input is already sigmoid(x)
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

// Pick index from probability distribution
int sample(const std::vector<float>& probs) {
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return probs.size() - 1;
}

int main() {
    srand(time(0));

    // 1. Load text
    std::ifstream file("text.txt");
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    
    if (text.empty()) {
        std::cerr << "text.txt is empty or missing.\n";
        return 1;
    }

    // 2. Build vocabulary
    std::map<char, int> char2idx;
    std::map<int, char> idx2char;
    for (char c : text) char2idx[c] = 0;
    int id = 0;
    for (auto& [ch, _] : char2idx) {
        char2idx[ch] = id;
        idx2char[id] = ch;
        id++;
    }
    int vocab_size = char2idx.size();

    // 3. Initialize weights
    std::vector<std::vector<float>> W1(vocab_size, std::vector<float>(EMBED_DIM));
    std::vector<std::vector<float>> W2(EMBED_DIM, std::vector<float>(HIDDEN_DIM));
    std::vector<std::vector<float>> W3(HIDDEN_DIM, std::vector<float>(vocab_size));

    for (auto& row : W1) for (float& val : row) val = randf();
    for (auto& row : W2) for (float& val : row) val = randf();
    for (auto& row : W3) for (float& val : row) val = randf();

    // 4. Training
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float loss = 0;

        for (size_t i = 0; i < text.size() - 1; ++i) {
            int x = char2idx[text[i]];
            int y = char2idx[text[i + 1]];

            // Forward pass
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

            // Backward pass
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

        std::cout << "Epoch " << epoch << ", Loss: " << loss / text.size() << "\n";
    }

    // 5. Generate text
    std::cout << "\nGenerated text:\n";
    int idx = char2idx[text[0]];
    for (int i = 0; i < 300; ++i) {
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
        std::cout << idx2char[idx];
    }

    std::cout << "\n";
    return 0;
}

