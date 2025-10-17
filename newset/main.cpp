// big_qa_nn.cpp
// Pure C++ (C++17) single-file large neural network for question ||| answer training.
// Author: ChatGPT (custom implementation)
// Usage:
//   Compile: g++ -O2 -std=c++17 big_qa_nn.cpp -o big_qa_nn
//   Train:   ./big_qa_nn train data.txt <epochs> <learning_rate>
//   Infer:   ./big_qa_nn infer
//
// Data format (each line):
//   question ||| answer
//
// Notes:
// - Character-level one-hot encoding for ASCII 0..127 (128 symbols).
// - Fixed max question and answer lengths (pad/truncate).
// - Network maps flattened question one-hot -> flattened answer one-hot.
// - MSE loss, sigmoid activations, SGD (per-example).
// - Saves weights to "weights.txt" every 50 epochs and at end.
// - If "weights.txt" exists, program loads and continues training.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <map>
#include <chrono>
#include <algorithm>
using namespace std;
using namespace std;

// -------------------- CONFIGURATION (tweak these if memory is too high) --------------------
const int VOCAB = 128;                // ASCII 0..127
const int DEFAULT_MAX_Q = 256;        // max question length in characters
const int DEFAULT_MAX_A = 256;        // max answer length in characters
// Default huge hidden layers — reduce if you run out of memory.
const int H1 = 2048;                  // first hidden layer size
const int H2 = 2048;                  // second hidden layer size
const string WEIGHT_FILE = "weights.txt";
const int SAVE_EVERY = 50;            // save every N epochs
// ------------------------------------------------------------------------------------------

// Utility RNG
std::mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
float frand(float a=-0.5f, float b=0.5f) {
    std::uniform_real_distribution<float> d(a,b);
    return d(rng);
}

// Activation and derivative (sigmoid)
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
inline float sigmoid_prime(float y) { // y is sigmoid(x)
    return y * (1.0f - y);
}

// Layer struct (fully connected)
struct Layer {
    int in, out;
    vector<float> weights; // flattened: out * in, index [i * out + j] meaning from input i to out j
    vector<float> bias;    // size out
    // scratch storage for forward/backprop
    vector<float> outvals; // size out
    vector<float> invals;  // size in (only used to store last forward input for backprop convenience)
    Layer() : in(0), out(0) {}
    Layer(int in_, int out_) { init(in_, out_); }
    void init(int in_, int out_) {
        in = in_; out = out_;
        weights.assign(in * out, 0.0f);
        bias.assign(out, 0.0f);
        outvals.assign(out, 0.0f);
        invals.assign(in, 0.0f);
        // random init
        float scale = 0.5f / sqrt(max(1, in));
        for (int i=0;i<in*out;i++) weights[i] = frand(-scale, scale);
        for (int j=0;j<out;j++) bias[j] = 0.0f;
    }
};

// Neural Network (list of layers)
struct NeuralNet {
    vector<Layer> layers;
    float lr;
    NeuralNet() : lr(0.01f) {}
    // layerSizes contains sizes including input and output, e.g. {input, h1, h2, output}
    void init_from_sizes(const vector<int>& layerSizes, float learning_rate) {
        lr = learning_rate;
        layers.clear();
        for (size_t i=0;i+1<layerSizes.size();++i) {
            layers.emplace_back(layerSizes[i], layerSizes[i+1]);
        }
    }

    vector<float> forward(const vector<float>& input) {
        vector<float> curr = input;
        for (size_t li=0; li<layers.size(); ++li) {
            Layer &L = layers[li];
            // store input for backprop
            L.invals = curr;
            // compute L.outvals
            for (int j=0;j<L.out;++j) {
                float s = L.bias[j];
                int base = j; // weights are [i*out + j]
                for (int i=0;i<L.in;++i) {
                    s += curr[i] * L.weights[i * L.out + j];
                }
                L.outvals[j] = sigmoid(s);
            }
            curr = L.outvals;
        }
        return curr;
    }

    // backprop using MSE loss; target size == final layer out
    // updates weights in place (SGD per-example)
    float backprop(const vector<float>& target) {
        int Lcount = layers.size();
        vector<vector<float>> deltas(Lcount);
        // compute delta for last layer
        Layer &last = layers.back();
        deltas.back().assign(last.out, 0.0f);
        float loss = 0.0f;
        for (int j=0;j<last.out;++j) {
            float o = last.outvals[j];
            float t = target[j];
            float err = o - t;
            loss += err*err;
            deltas.back()[j] = err * sigmoid_prime(o);
        }
        // backpropagate deltas
        for (int li = Lcount-1; li > 0; --li) {
            Layer &L = layers[li];
            Layer &P = layers[li-1];
            deltas[li-1].assign(P.out, 0.0f);
            for (int i=0;i<P.out;++i) {
                float sum = 0.0f;
                for (int j=0;j<L.out;++j) {
                    sum += L.weights[i * L.out + j] * deltas[li][j];
                }
                float outp = P.outvals[i];
                deltas[li-1][i] = sum * sigmoid_prime(outp);
            }
        }
        // update weights and biases
        for (int li=0; li<Lcount; ++li) {
            Layer &L = layers[li];
            const vector<float> &invals = L.invals;
            for (int j=0;j<L.out;++j) {
                // bias update
                L.bias[j] -= lr * deltas[li][j];
                // weight update
                for (int i=0;i<L.in;++i) {
                    int idx = i * L.out + j;
                    L.weights[idx] -= lr * invals[i] * deltas[li][j];
                }
            }
        }
        return loss * 0.5f; // squared error / 2 per sample (sum over outputs)
    }

    // Save weights in a simple text format:
    // first line: layer_count
    // second line: sizes separated
    // then for each layer: all weights (in * out floats) then biases (out floats)
    bool save(const string &fn) {
        ofstream fout(fn, ios::binary);
        if (!fout) return false;
        int Lcount = layers.size();
        fout << Lcount << "\n";
        for (auto &L : layers) fout << L.in << " " << L.out << " ";
        fout << "\n";
        for (auto &L : layers) {
            // weights
            for (float w : L.weights) fout << std::setprecision(9) << w << " ";
            // biases
            for (float b : L.bias) fout << std::setprecision(9) << b << " ";
            fout << "\n";
        }
        fout.close();
        return true;
    }

    bool load(const string &fn) {
        ifstream fin(fn);
        if (!fin) return false;
        int Lcount;
        fin >> Lcount;
        vector<int> ins(Lcount), outs(Lcount);
        for (int i=0;i<Lcount;i++) {
            fin >> ins[i] >> outs[i];
        }
        // reconstruct layers container
        layers.clear();
        for (int i=0;i<Lcount;i++) {
            Layer L;
            L.init(ins[i], outs[i]);
            // read weights
            for (int k=0;k<L.in * L.out; ++k) {
                float w; fin >> w; L.weights[k] = w;
            }
            for (int j=0;j<L.out;++j) {
                float b; fin >> b; L.bias[j] = b;
            }
            layers.push_back(std::move(L));
        }
        fin.close();
        return true;
    }

    int total_params() const {
        int64_t sum = 0;
        for (auto &L : layers) sum += (int64_t)L.in * L.out + L.out;
        return (int)sum;
    }
};

// -------------------- Text encoding helpers --------------------
// Map chars 0..127 directly. Anything outside gets set to 32 (space).
inline int char_to_idx(char c) {
    unsigned char uc = (unsigned char)c;
    if (uc < VOCAB) return (int)uc;
    return 32;
}
inline char idx_to_char(int idx) {
    if (idx < 0 || idx >= VOCAB) return ' ';
    return (char)idx;
}

// Flattened one-hot encoding: inputSize = maxLen * VOCAB
vector<float> encode_string(const string &s, int maxLen) {
    vector<float> v(maxLen * VOCAB, 0.0f);
    int L = (int)min((int)s.size(), maxLen);
    for (int i=0;i<L;++i) {
        int idx = char_to_idx(s[i]);
        v[i * VOCAB + idx] = 1.0f;
    }
    // pad remainder as all-zero (equivalent to space omitted)
    return v;
}

string decode_flat_output(const vector<float> &out, int maxLen) {
    // out length should be maxLen * VOCAB
    string s;
    for (int i=0;i<maxLen;++i) {
        int best = 0;
        float bestv = out[i*VOCAB + 0];
        for (int k=1;k<VOCAB;++k) {
            float val = out[i*VOCAB + k];
            if (val > bestv) { bestv = val; best = k; }
        }
        char c = idx_to_char(best);
        s.push_back(c);
    }
    // trim trailing spaces
    while (!s.empty() && s.back()==' ') s.pop_back();
    return s;
}

// -------------------- Dataset loader --------------------
struct Example {
    string q, a;
    vector<float> in;  // encoded question
    vector<float> out; // encoded answer
};

vector<Example> load_dataset(const string &filename, int maxQ, int maxA) {
    vector<Example> data;
    ifstream fin(filename);
    if (!fin) {
        cerr << "Could not open data file: " << filename << "\n";
        return data;
    }
    string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        // find "|||"
        size_t pos = line.find("|||");
        if (pos == string::npos) continue;
        string q = line.substr(0, pos);
        // trim spaces around q
        while (!q.empty() && isspace((unsigned char)q.back())) q.pop_back();
        while (!q.empty() && isspace((unsigned char)q.front())) q.erase(q.begin());
        string a = line.substr(pos + 3);
        while (!a.empty() && isspace((unsigned char)a.front())) a.erase(a.begin());
        while (!a.empty() && isspace((unsigned char)a.back())) a.pop_back();
        // truncate or pad
        if ((int)q.size() > maxQ) q = q.substr(0, maxQ);
        if ((int)a.size() > maxA) a = a.substr(0, maxA);
        Example e;
        e.q = q;
        e.a = a;
        e.in = encode_string(q, maxQ);
        // encode answer into flattened one-hot vector: length = maxA * VOCAB
        e.out.assign(maxA * VOCAB, 0.0f);
        for (int i=0;i<(int)a.size();++i) {
            int idx = char_to_idx(a[i]);
            e.out[i * VOCAB + idx] = 1.0f;
        }
        data.push_back(std::move(e));
    }
    fin.close();
    cout << "Loaded " << data.size() << " examples from " << filename << "\n";
    return data;
}

// -------------------- Training / CLI --------------------

void train_mode(const string &datafile, int epochs, float lr,
                int maxQ=DEFAULT_MAX_Q, int maxA=DEFAULT_MAX_A) {

    // Load dataset
    auto data = load_dataset(datafile, maxQ, maxA);
    if (data.empty()) {
        cerr << "No data -> exiting.\n"; return;
    }

    // Define layer sizes: input = maxQ * VOCAB, output = maxA * VOCAB
    int inputSize = maxQ * VOCAB;
    int outputSize = maxA * VOCAB;

    // Build architecture vector
    vector<int> sizes;
    sizes.push_back(inputSize);
    sizes.push_back(H1); // large hidden
    sizes.push_back(H2); // large hidden
    sizes.push_back(outputSize);

    NeuralNet net;
    // If weights file exists, try to load; otherwise initialize new network
    ifstream ifs(WEIGHT_FILE);
    if (ifs.good()) {
        ifs.close();
        cout << "Found weights file '"<<WEIGHT_FILE<<"'. Attempting to load and continue training.\n";
        bool okload = net.load(WEIGHT_FILE);
        if (!okload) {
            cerr << "Failed to load weights. Initializing new network.\n";
            net.init_from_sizes(sizes, lr);
        } else {
            // basic check: loaded network sizes should match expected
            if ((int)net.layers.front().in != inputSize || (int)net.layers.back().out != outputSize) {
                cerr << "Loaded network sizes do not match dataset encoding. Re-initializing network.\n";
                net.init_from_sizes(sizes, lr);
            } else {
                net.lr = lr;
            }
        }
    } else {
        cout << "No weights file found. Initializing new large network.\n";
        net.init_from_sizes(sizes, lr);
    }

    cout << "Network params (approx): " << net.total_params() << " parameters\n";
    cout << "Training for " << epochs << " epochs, learning rate " << lr << ". Checkpoint every " << SAVE_EVERY << " epochs.\n";

    // Training loop (simple SGD, per-example)
    for (int epoch=1; epoch<=epochs; ++epoch) {
        double epoch_loss = 0.0;
        // shuffle data every epoch
        std::shuffle(data.begin(), data.end(), rng);
        for (size_t ei=0; ei<data.size(); ++ei) {
            // forward
            vector<float> out = net.forward(data[ei].in);
            // compute loss and backprop
            float sample_loss = net.backprop(data[ei].out);
            epoch_loss += sample_loss;
        }
        epoch_loss /= max(1,(int)data.size());
        cout << "Epoch " << epoch << " | Avg Loss: " << std::fixed << std::setprecision(6) << epoch_loss << "\n";
        if (epoch % SAVE_EVERY == 0 || epoch == epochs) {
            cout << "Saving weights to " << WEIGHT_FILE << " (epoch " << epoch << ")\n";
            bool ok = net.save(WEIGHT_FILE);
            if (!ok) cerr << "Warning: failed to save weights to " << WEIGHT_FILE << "\n";
            // Also show a few sample outputs for quick feedback
            cout << "=== Sample outputs ===\n";
            for (int s=0; s<min(3, (int)data.size()); ++s) {
                int idx = (epoch + s) % data.size();
                auto &ex = data[idx];
                vector<float> pred = net.forward(ex.in);
                string dec = decode_flat_output(pred, maxA);
                cout << "Q: " << ex.q << "\nA(truth): " << ex.a << "\nA(pred) : " << dec << "\n---\n";
            }
        }
    }
    cout << "Training complete. Final save.\n";
    net.save(WEIGHT_FILE);
    cout << "Saved to " << WEIGHT_FILE << "\n";
}

void infer_mode(int maxQ=DEFAULT_MAX_Q, int maxA=DEFAULT_MAX_A) {
    // load weights
    NeuralNet net;
    if (!net.load(WEIGHT_FILE)) {
        cerr << "Failed to load weights file '" << WEIGHT_FILE << "'. Train a model first.\n";
        return;
    }
    cout << "Loaded network. Total params ~ " << net.total_params() << "\n";
    cout << "Entering interactive inference. Type a question and press enter. Empty line to quit.\n";
    string line;
    while (true) {
        cout << "\n> ";
        if (!std::getline(cin, line)) break;
        if (line.size() == 0) break;
        string q = line;
        if ((int)q.size() > maxQ) q = q.substr(0, maxQ);
        auto in = encode_string(q, maxQ);
        vector<float> out = net.forward(in);
        string ans = decode_flat_output(out, maxA);
        cout << "-> " << ans << "\n";
    }
}

// Simple CLI for train / infer
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cout << "Usage:\n  " << argv[0] << " train data.txt <epochs> <learning_rate>\n";
        cout << "  " << argv[0] << " infer\n\n";
        cout << "Config (defaults):\n";
        cout << "  VOCAB="<<VOCAB<<", MAX_Q="<<DEFAULT_MAX_Q<<", MAX_A="<<DEFAULT_MAX_A<<"\n";
        cout << "  Hidden sizes = " << H1 << ", " << H2 << "\n";
        cout << "Warning: large settings use substantial memory. Adjust constants near top of the file if needed.\n";
        return 0;
    }

    string mode = argv[1];
    if (mode == "train") {
        if (argc < 3) { cerr << "train requires data file\n"; return 1; }
        string datafile = argv[2];
        int epochs = 100;
        float lr = 0.01f;
        if (argc >= 4) epochs = stoi(argv[3]);
        if (argc >= 5) lr = stof(argv[4]);
        train_mode(datafile, epochs, lr, DEFAULT_MAX_Q, DEFAULT_MAX_A);
    } else if (mode == "infer") {
        infer_mode(DEFAULT_MAX_Q, DEFAULT_MAX_A);
    } else {
        cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}