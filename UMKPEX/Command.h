
class CommandLine
{
public:

	string branching, bounding, algorithm, ctcp, heuristic, framework;
	string race; // raw -race spec (parsed by parse_race_options in main.cpp); empty = no racing
	bool UBR2;
	// bool BR1, BR2, RR1, RR2, RR3;

	// Valid-value lists, shared between this constructor's own validation below and
	// main.cpp's parse_race_options (which validates per-option overrides for
	// -race using the same lists, so a race option can vary any of these fields
	// independently of the others).
	static vector<string> valid_algorithms() { return {"UMKP+", "kPEX"}; }
	static vector<string> valid_branchings() { return {"Adaptive-Br", "Pivot-Br", "S-Br", "R-Br", "SR-Br", "Binary-Br", "BinaryM-Br"}; }
	static vector<string> valid_boundings() { return {"None", "S-Bound", "R-Bound", "SR-Bound", "AltRB"}; }
	static vector<string> valid_heuristics() { return {"kPEX-Heur", "EGO-Heur", "Both-Heur"}; }
	static vector<string> valid_ctcp_methods() { return {"CTCP", "CF-CTCP"}; }
	static vector<string> valid_frameworks() { return {"default", "oriented"}; }



  CommandLine(int _argc, char **_argv) : argc(_argc), argv(_argv)
  {
    algorithm = GetOptionValue("-algorithm", "kPEX");
    branching = GetOptionValue("-branching", "None");
    bounding = GetOptionValue("-bounding", "AltRB");
    UBR2 = GetOptionValue("-UBR2", "true") == "true";
    ctcp = GetOptionValue("-ctcp", "CTCP");
    heuristic = GetOptionValue("-heuristic", "Both-Heur");
    framework = GetOptionValue("-framework", "default");
    race = GetOptionValue("-race", "");


    cout<<"----------------------Parameters-----------------------\n";
    vector<string> algos=valid_algorithms();
    if(find(algos.begin(), algos.end(), algorithm)==algos.end()) {
      cout<<"Not a valid selection for algorithm, using 'kPEX'"<<endl;
      cout<<"Valid algorithms: "; for(auto& b: algos) cout<<b<<" ";
      cout<<endl;
      algorithm = "kPEX";
    }
    vector<string> branchings=valid_branchings();
    if(find(branchings.begin(), branchings.end(), branching)==branchings.end()) {
      cout<<"Valid branching methods: "; for(auto& b: branchings) cout<<b<<" ";
      cout<<endl;
      if(algorithm=="kPEX"){
        cout<<"Not a valid selection for branching, using Adaptive-Br"<<endl;
        branching="Adaptive-Br";
      }
      else{
        cout<<"Not a valid selection for branching, using Adaptive-Br"<<endl;
        branching = "Adaptive-Br";
      }
    }

    vector<string> boundings=valid_boundings();
    if(find(boundings.begin(), boundings.end(), bounding)==boundings.end()) {
      cout<<"Not a valid selection for bounding, using 'AltRB'"<<endl;
      cout<<"Valid bounding methods: "; for(auto& b: boundings) cout<<b<<" ";
      cout<<endl;
      bounding = "AltRB";
    }

    vector<string> heuristics=valid_heuristics();
    if(find(heuristics.begin(), heuristics.end(), heuristic)==heuristics.end()) {
      cout<<"Not a valid selection for Heuristic, using 'Both-Heur'"<<endl;
      cout<<"Valid Heuristic methods: "; for(auto& b: heuristics) cout<<b<<" ";
      cout<<endl;
      heuristic = "Both-Heur";
    }

    vector<string> ctcp_methods=valid_ctcp_methods();
    if(find(ctcp_methods.begin(), ctcp_methods.end(), ctcp)==ctcp_methods.end()) {
      cout<<"Valid CTCP methods: "; for(auto& b: ctcp_methods) cout<<b<<" ";
      cout<<endl;
      cout<<"Using 'CF-CTCP'"<<endl;
      ctcp = "CF-CTCP";
    }

    vector<string> frameworks=valid_frameworks();
    if(find(frameworks.begin(), frameworks.end(), framework)==frameworks.end()) {
      cout<<"Valid frameworks: "; for(auto& b: frameworks) cout<<b<<" ";
      cout<<endl;
      cout<<"Using 'default'"<<endl;
      framework = "default";
    }
    cout<<"Dataset: "<<argv[1]<<endl;
    cout<<"Algorithm: "<<algorithm<<endl;
    cout<<"K: "<<argv[2]<<endl;
    cout<<"Branching: "<<branching<<endl;
    cout<<"Bounding: "<<bounding<<endl;
    cout<<"UBR2: "<<UBR2<<endl;
    cout<<"CTCP Method: "<<ctcp<<endl;
    cout<<"Heuristic: "<<heuristic<<endl;
    cout<<"Framework: "<<framework<<endl;
    if(!race.empty())
      cout<<"Race: "<<race<<endl;
    cout<<"----------------------**********-----------------------\n";
  }

  CommandLine() : argc(0), argv(nullptr) {}
  void BadArgument()
  {
    std::cout << "usage: " << argv[0] << " bad argument" << std::endl;
    abort();
  }

  char *GetOptionValue(const std::string &option)
  {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
        return argv[i + 1];
    return NULL;
  }
  bool GetOptionEnabled(const std::string &option)
  {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
        return true;
    return false;
  }
  std::string GetOptionValue(const std::string &option, std::string defaultValue)
  {

    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
        return (std::string)argv[i + 1];
    return defaultValue;
  }

  int GetOptionIntValue(const std::string &option, int defaultValue)
  {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
      {
        int r = atoi(argv[i + 1]);
        return r;
      }
    return defaultValue;
  }

  long GetOptionLongValue(const std::string &option, long defaultValue)
  {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
      {
        long r = atol(argv[i + 1]);
        return r;
      }
    return defaultValue;
  }

  double GetOptionDoubleValue(const std::string &option, double defaultValue)
  {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
      {
        double val;
        if (sscanf(argv[i + 1], "%lf", &val) == EOF)
        {
          BadArgument();
        }
        return val;
      }
    return defaultValue;
  }

  int argc;
  char **argv;
};

// thread_local: each racer thread (see run_parallel_bnb in main.cpp) gets its own
// independent copy, so overriding e.g. PARA.framework in one thread can't race with
// another thread's copy (PARA.branching is also mutated mid-search, see Branch.h).
thread_local CommandLine PARA;
