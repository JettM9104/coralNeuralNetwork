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

const int EMBED_DIM = 16;    // smaller embedding size
const int HIDDEN_DIM = 32;   // smaller hidden layer
const int EPOCHS = 100;      // limit training epochs
const int BATCH_SIZE = 8;
const float LEARNING_RATE = 0.05f;
const size_t MAX_ANSWER_LEN = 8;   // limit answer length used for training
const size_t MAX_QA_PAIRS = 200;   // limit total QA pairs loaded for training

float randf() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float dsigmoid(float y) {
    return y * (1 - y);
}

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

void update_weights_batch(
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
        for (int j = 0; j < vocab_size; ++j) {
            for (int k = 0; k < HIDDEN_DIM; ++k)
                logits[j] += hidden[k] * W3[k][j];
        }

        auto probs = softmax(logits);

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
}

int main() {
    srand(time(0));
    std::mt19937 rng(time(0));

    std::ifstream file("text.txt");
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        std::cerr << "text.txt is empty or missing.\n";
        return 1;
    }

    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> qa_pairs;
    std::map<std::string, int> word2idx;
    std::map<int, std::string> idx2word;
    int id = 0;

    word2idx["<eos>"] = id;
    idx2word[id++] = "<eos>";

    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        size_t sep = line.find("|||");
        if (sep == std::string::npos) continue;

        std::string question = line.substr(0, sep);
        std::string answer = line.substr(sep + 3);

        auto q_words = tokenize(question);
        auto a_words = tokenize(answer);
        a_words.push_back("<eos>");  // Keep <eos> to mark answer end

        qa_pairs.push_back({q_words, a_words});

        for (auto& w : q_words)
            if (!word2idx.count(w)) { word2idx[w] = id; idx2word[id++] = w; }
        for (auto& w : a_words)
            if (!word2idx.count(w)) { word2idx[w] = id; idx2word[id++] = w; }
    }

    // Limit training data size for speed
    if (qa_pairs.size() > MAX_QA_PAIRS)
        qa_pairs.resize(MAX_QA_PAIRS);

    int vocab_size = word2idx.size();

    std::vector<std::vector<float>> W1(vocab_size, std::vector<float>(EMBED_DIM));
    std::vector<std::vector<float>> W2(EMBED_DIM, std::vector<float>(HIDDEN_DIM));
    std::vector<std::vector<float>> W3(HIDDEN_DIM, std::vector<float>(vocab_size));

    // Initialize or load weights
    std::ifstream w1f("W1.txt");
    if (w1f.good()) {
        load_matrix("W1.txt", W1);
        load_matrix("W2.txt", W2);
        load_matrix("W3.txt", W3);
    }
    else {
        for (auto& row : W1)
            for (float& val : row)
                val = randf() * 0.1f;
        for (auto& row : W2)
            for (float& val : row)
                val = randf() * 0.1f;
        for (auto& row : W3)
            for (float& val : row)
                val = randf() * 0.1f;
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::vector<int> batch_x, batch_y;

        for (const auto& [q, a] : qa_pairs) {
            // Use only first MAX_ANSWER_LEN tokens for training answer
            size_t len = std::min(a.size(), MAX_ANSWER_LEN);

            for (size_t i = 0; i + 1 < len; ++i) {
                batch_x.push_back(word2idx[a[i]]);
                batch_y.push_back(word2idx[a[i + 1]]);

                if (batch_x.size() >= BATCH_SIZE) {
                    update_weights_batch(W1, W2, W3, batch_x, batch_y, vocab_size);
                    batch_x.clear();
                    batch_y.clear();
                }
            }
        }

        if (!batch_x.empty()) {
            update_weights_batch(W1, W2, W3, batch_x, batch_y, vocab_size);
        }

        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << " complete.\n";
    }

    save_matrix("W1.txt", W1);
    save_matrix("W2.txt", W2);
    save_matrix("W3.txt", W3);

    return 0;
}
