compile() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: compile [<filename>] [flags]"
    echo "  Compile a C++ source file."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  compile                   # compiles main.cpp"
    echo "  compile -O3 -Wall         # compiles main.cpp with flags"
    echo "  compile test.cpp -O2      # compiles test.cpp with flags"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  clang++ "$filename" -o "${filename%.*}" "${flags[@]}"
}

compile11() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: compile11 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++11."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  compile11                   # compiles main.cpp"
    echo "  compile11 -O3 -Wall         # compiles main.cpp with flags"
    echo "  compile11 test.cpp -O2      # compiles test.cpp with flags"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  clang++ "$filename" -o "${filename%.*}" "${flags[@]}" "-std=c++11"
}

compile14() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: compile14 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++14."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  compile14                   # compiles main.cpp"
    echo "  compile14 -O3 -Wall         # compiles main.cpp with flags"
    echo "  compile14 test.cpp -O2      # compiles test.cpp with flags"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  clang++ "$filename" -o "${filename%.*}" "${flags[@]}" "-std=c++14"
}

compile17() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: compile17 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++17."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  compile17                   # compiles main.cpp"
    echo "  compile17 -O3 -Wall         # compiles main.cpp with flags"
    echo "  compile17 test.cpp -O2      # compiles test.cpp with flags"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  clang++ "$filename" -o "${filename%.*}" "${flags[@]}" "-std=c++17"
}

compile20() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: compile20 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++20."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  compile20                   # compiles main.cpp"
    echo "  compile20 -O3 -Wall         # compiles main.cpp with flags"
    echo "  compile20 test.cpp -O2      # compiles test.cpp with flags"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  clang++ "$filename" -o "${filename%.*}" "${flags[@]}" "-std=c++20"
}

compile23() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: compile23 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++23."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  compile23                   # compiles main.cpp"
    echo "  compile23 -O3 -Wall         # compiles main.cpp with flags"
    echo "  compile23 test.cpp -O2      # compiles test.cpp with flags"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  clang++ "$filename" -o "${filename%.*}" "${flags[@]}" "-std=c++23"
}

run() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: run [<executable_name>]"
    echo "  Run a compiled executable."
    echo "  - If no executable name is given, defaults to ./main"
    echo "Examples:"
    echo "  run            # runs ./main"
    echo "  run myprogram  # runs ./myprogram"
    return 0
  fi

  local executable="${1:-main}"
  ./"$executable"
}

car() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: car [<filename>] [flags]"
    echo "  Compile a C++ source file and run the resulting executable."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  car               # compile and run main.cpp"
    echo "  car -O3           # compile main.cpp with flags and run"
    echo "  car test.cpp -O2  # compile test.cpp with flags and run"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  compile "$filename" "${flags[@]}"
  if [ $? -eq 0 ]; then
    run "${filename%.*}"
  else
    echo "Compilation failed."
  fi
}


car11() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: car11 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++11 and run the resulting executable."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  car11               # compile and run main.cpp"
    echo "  car11 -O3           # compile main.cpp with flags and run"
    echo "  car11 test.cpp -O2  # compile test.cpp with flags and run"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  compile11 "$filename" "${flags[@]}"
  if [ $? -eq 0 ]; then
    run "${filename%.*}"
  else
    echo "Compilation failed."
  fi
}


car14() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: car14 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++14 and run the resulting executable."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  car14               # compile and run main.cpp"
    echo "  car14 -O3           # compile main.cpp with flags and run"
    echo "  car14 test.cpp -O2  # compile test.cpp with flags and run"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  compile14 "$filename" "${flags[@]}"
  if [ $? -eq 0 ]; then
    run "${filename%.*}"
  else
    echo "Compilation failed."
  fi
}


car17() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: car17 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++17 and run the resulting executable."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  car17               # compile and run main.cpp"
    echo "  car17 -O3           # compile main.cpp with flags and run"
    echo "  car17 test.cpp -O2  # compile test.cpp with flags and run"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  compile17 "$filename" "${flags[@]}"
  if [ $? -eq 0 ]; then
    run "${filename%.*}"
  else
    echo "Compilation failed."
  fi
}


car20() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: car20 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++20 and run the resulting executable."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  car20               # compile and run main.cpp"
    echo "  car20 -O3           # compile main.cpp with flags and run"
    echo "  car20 test.cpp -O2  # compile test.cpp with flags and run"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  compile20 "$filename" "${flags[@]}"
  if [ $? -eq 0 ]; then
    run "${filename%.*}"
  else
    echo "Compilation failed."
  fi
}


car23() {
  if [[ "$1" == "--help" ]]; then
    echo "Usage: car23 [<filename>] [flags]"
    echo "  Compile a C++ source file with C++23 and run the resulting executable."
    echo "  - If no filename is given, defaults to main.cpp"
    echo "  - If first argument starts with '-', it's treated as flags for main.cpp"
    echo "Examples:"
    echo "  car23               # compile and run main.cpp"
    echo "  car23 -O3           # compile main.cpp with flags and run"
    echo "  car23 test.cpp -O2  # compile test.cpp with flags and run"
    return 0
  fi

  if [ $# -eq 0 ]; then
    local filename="main.cpp"
    local flags=()
  else
    if [[ "$1" == -* ]]; then
      local filename="main.cpp"
      local flags=("$@")
    else
      local filename="$1"
      shift
      local flags=("$@")
    fi
  fi

  compile23 "$filename" "${flags[@]}"
  if [ $? -eq 0 ]; then
    run "${filename%.*}"
  else
    echo "Compilation failed."
  fi
}

alias ip='echo "Private Local IPv4 (Wi-Fi): $(ipconfig getifaddr en0)"; echo "Public IPv4: $(curl -s ifconfig.me)"; echo "Interface Details:"; networksetup -getinfo Wi-Fi'