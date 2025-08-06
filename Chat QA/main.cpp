#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <cstring>
#include <sys/stat.h>

// Hyperparameters
const int EMBEDDING_SIZE = 16;
const int HIDDEN_SIZE = 32;
const int EPOCHS = 100'000;
const float LEARNING_RATE = 0.01;

// Types
using Vocab = std::unordered_map<std::string, int>;
using ReverseVocab = std::vector<std::string>;

// Activation
float relu(float x) { return x > 0 ? x : 0; }
float relu_deriv(float x) { return x > 0 ? 1 : 0; }

// Softmax
std::vector<float> softmax(const std::vector<float>& x) {
    float max_elem = *std::max_element(x.begin(), x.end());
    std::vector<float> exp_x(x.size());
    float sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = std::exp(x[i] - max_elem);
        sum += exp_x[i];
    }
    for (float& val : exp_x) val /= sum;
    return exp_x;
}

// Tokenizer
std::vector<std::string> tokenize(const std::string& line) {
    std::stringstream ss(line);
    std::string word;
    std::vector<std::string> tokens;
    while (ss >> word) tokens.push_back(word);
    return tokens;
}

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Neural Net
struct MLP {
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<float>> w1, w2;
    std::vector<float> b1, b2;
    int vocab_size, output_size;

    MLP(int vocab_size, int output_size) : vocab_size(vocab_size), output_size(output_size) {
        auto randn = []() {
            return ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        };

        embeddings = std::vector<std::vector<float>>(vocab_size, std::vector<float>(EMBEDDING_SIZE));
        for (auto& e : embeddings) for (auto& v : e) v = randn();

        w1 = std::vector<std::vector<float>>(HIDDEN_SIZE, std::vector<float>(EMBEDDING_SIZE));
        b1 = std::vector<float>(HIDDEN_SIZE);
        for (auto& row : w1) for (auto& val : row) val = randn();

        w2 = std::vector<std::vector<float>>(output_size, std::vector<float>(HIDDEN_SIZE));
        b2 = std::vector<float>(output_size);
        for (auto& row : w2) for (auto& val : row) val = randn();
    }

    std::vector<float> forward(const std::vector<int>& input) {
        std::vector<float> avg(EMBEDDING_SIZE, 0);
        for (int idx : input)
            for (int i = 0; i < EMBEDDING_SIZE; ++i)
                avg[i] += embeddings[idx][i];
        for (float& x : avg) x /= input.size();

        std::vector<float> h(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            h[i] = b1[i];
            for (int j = 0; j < EMBEDDING_SIZE; ++j)
                h[i] += w1[i][j] * avg[j];
            h[i] = relu(h[i]);
        }

        std::vector<float> out(output_size);
        for (int i = 0; i < output_size; ++i) {
            out[i] = b2[i];
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                out[i] += w2[i][j] * h[j];
        }
        return softmax(out);
    }

    void train(const std::vector<std::pair<std::vector<int>, int>>& data) {
        for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
            float loss = 0;
            for (const auto& [input, target] : data) {
                std::vector<float> avg(EMBEDDING_SIZE, 0);
                for (int idx : input)
                    for (int i = 0; i < EMBEDDING_SIZE; ++i)
                        avg[i] += embeddings[idx][i];
                for (float& x : avg) x /= input.size();

                std::vector<float> h(HIDDEN_SIZE), dh(HIDDEN_SIZE);
                for (int i = 0; i < HIDDEN_SIZE; ++i) {
                    h[i] = b1[i];
                    for (int j = 0; j < EMBEDDING_SIZE; ++j)
                        h[i] += w1[i][j] * avg[j];
                    h[i] = relu(h[i]);
                }

                std::vector<float> out(output_size);
                for (int i = 0; i < output_size; ++i) {
                    out[i] = b2[i];
                    for (int j = 0; j < HIDDEN_SIZE; ++j)
                        out[i] += w2[i][j] * h[j];
                }

                auto probs = softmax(out);
                loss -= std::log(probs[target] + 1e-6);

                std::vector<float> dlogit(output_size);
                for (int i = 0; i < output_size; ++i)
                    dlogit[i] = probs[i];
                dlogit[target] -= 1.0f;

                for (int i = 0; i < output_size; ++i)
                    dlogit[i] *= probs[i];

                std::vector<float> dhidden(HIDDEN_SIZE, 0);
                for (int i = 0; i < output_size; ++i)
                    for (int j = 0; j < HIDDEN_SIZE; ++j)
                        dhidden[j] += w2[i][j] * dlogit[i];

                for (int i = 0; i < output_size; ++i)
                    for (int j = 0; j < HIDDEN_SIZE; ++j)
                        w2[i][j] -= LEARNING_RATE * dlogit[i] * h[j];
                for (int i = 0; i < output_size; ++i)
                    b2[i] -= LEARNING_RATE * dlogit[i];

                for (int i = 0; i < HIDDEN_SIZE; ++i)
                    if (h[i] > 0)
                        for (int j = 0; j < EMBEDDING_SIZE; ++j)
                            w1[i][j] -= LEARNING_RATE * dhidden[i] * avg[j];
                for (int i = 0; i < HIDDEN_SIZE; ++i)
                    b1[i] -= LEARNING_RATE * dhidden[i] * (h[i] > 0);
            }

            if (epoch % 25 == 0)
                std::cout << "Epoch " << epoch << " Loss: " << loss / data.size() << "\n";
        }
    }

    void save(const std::string& filename, const Vocab& vocab, const ReverseVocab& rev) {
        std::ofstream f(filename, std::ios::binary);
        int vs = vocab.size();
        f.write((char*)&vs, sizeof(int));
        for (auto& [word, idx] : vocab) {
            int len = word.size();
            f.write((char*)&len, sizeof(int));
            f.write(word.c_str(), len);
        }
        for (auto& v : embeddings) f.write((char*)v.data(), sizeof(float) * EMBEDDING_SIZE);
        for (auto& v : w1) f.write((char*)v.data(), sizeof(float) * EMBEDDING_SIZE);
        f.write((char*)b1.data(), sizeof(float) * HIDDEN_SIZE);
        for (auto& v : w2) f.write((char*)v.data(), sizeof(float) * HIDDEN_SIZE);
        f.write((char*)b2.data(), sizeof(float) * output_size);
        f.close();
    }

    void load(const std::string& filename, Vocab& vocab, ReverseVocab& rev) {
        std::ifstream f(filename, std::ios::binary);
        int vs;
        f.read((char*)&vs, sizeof(int));
        for (int i = 0; i < vs; ++i) {
            int len;
            f.read((char*)&len, sizeof(int));
            std::string word(len, 0);
            f.read(&word[0], len);
            vocab[word] = i;
            rev.push_back(word);
        }
        embeddings.resize(vs, std::vector<float>(EMBEDDING_SIZE));
        for (auto& v : embeddings) f.read((char*)v.data(), sizeof(float) * EMBEDDING_SIZE);
        for (auto& v : w1) f.read((char*)v.data(), sizeof(float) * EMBEDDING_SIZE);
        f.read((char*)b1.data(), sizeof(float) * HIDDEN_SIZE);
        for (auto& v : w2) f.read((char*)v.data(), sizeof(float) * HIDDEN_SIZE);
        f.read((char*)b2.data(), sizeof(float) * output_size);
        f.close();
    }
};

int main() {
    std::ifstream fin("train.txt");
    std::string line;
    Vocab vocab;
    ReverseVocab rev;
    int idx = 0;
    std::vector<std::pair<std::vector<int>, int>> data;

    while (getline(fin, line)) {
        auto sep = line.find("|||");
        if (sep == std::string::npos) continue;
        std::string q = line.substr(0, sep);
        std::string a = line.substr(sep + 3);

        auto qtokens = tokenize(q);
        auto atokens = tokenize(a);

        std::vector<int> input;
        for (const std::string& tok : qtokens) {
            if (!vocab.count(tok)) {
                vocab[tok] = idx++;
                rev.push_back(tok);
            }
            input.push_back(vocab[tok]);
        }

        std::string atok = atokens[0];
        if (!vocab.count(atok)) {
            vocab[atok] = idx++;
            rev.push_back(atok);
        }
        int target = vocab[atok];

        data.push_back({input, target});
    }

    MLP model(vocab.size(), vocab.size());
    if (file_exists("model.bin")) {
        std::cout << "Loading existing model...\n";
        model.load("model.bin", vocab, rev);
    } else {
        std::cout << "Training from scratch...\n";
        model.train(data);
        model.save("model.bin", vocab, rev);
    }

    // Chat loop
    std::cout << "Chat mode! Type a question:\n";
    while (true) {
        std::string in;
        std::getline(std::cin, in);
        auto toks = tokenize(in);
        std::vector<int> input;
        for (auto& t : toks) {
            if (!vocab.count(t)) {
                std::cout << "(unknown word: " << t << ")\n";
                continue;
            }
            input.push_back(vocab[t]);
        }
        auto out = model.forward(input);
        int best = std::max_element(out.begin(), out.end()) - out.begin();
        std::cout << "🧠: " << rev[best] << "\n";
    }

    return 0;
}