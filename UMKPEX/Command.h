
class CommandLine
{
public:

	string branching, bounding, algorithm, ctcp, heuristic;
	bool UBR2;
	// bool BR1, BR2, RR1, RR2, RR3;



  CommandLine(int _argc, char **_argv) : argc(_argc), argv(_argv)
  {
    algorithm = GetOptionValue("-algorithm", "kPEX");
    branching = GetOptionValue("-branching", "None");
    bounding = GetOptionValue("-bounding", "AltRB");
    UBR2 = GetOptionValue("-UBR2", "true") == "true";
    ctcp = GetOptionValue("-ctcp", "CF-CTCP");
    heuristic = GetOptionValue("-heuristic", "kPEX-Heur");


    cout<<"----------------------Parameters-----------------------\n";
    vector<string> algos={"UMKP+", "kPEX"};
    if(find(algos.begin(), algos.end(), algorithm)==algos.end()) {
      cout<<"Not a valid selection for algorithm, using 'kPEX'"<<endl;
      cout<<"Valid algorithms: "; for(auto& b: algos) cout<<b<<" ";
      cout<<endl;
      algorithm = "kPEX";
    }
    vector<string> branchings={"Adaptive-Br", "Pivot-Br", "S-Br", "R-Br", "SR-Br", "Binary-Br"};
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

    vector<string> boundings={"None", "S-Bound", "R-Bound", "SR-Bound", "AltRB"};
    if(find(boundings.begin(), boundings.end(), bounding)==boundings.end()) {
      cout<<"Not a valid selection for bounding, using 'AltRB'"<<endl;
      cout<<"Valid bounding methods: "; for(auto& b: boundings) cout<<b<<" ";
      cout<<endl;
      bounding = "AltRB";
    }

    vector<string> heuristics={"kPEX-Heur", "EGO-Heur", "Both-Heur"};
    if(find(heuristics.begin(), heuristics.end(), heuristic)==heuristics.end()) {
      cout<<"Not a valid selection for Heuristic, using 'Both-Heur'"<<endl;
      cout<<"Valid Heuristic methods: "; for(auto& b: heuristics) cout<<b<<" ";
      cout<<endl;
      heuristic = "Both-Heur";
    }

    vector<string> ctcp_methods={"CTCP", "CF-CTCP"};
    if(find(ctcp_methods.begin(), ctcp_methods.end(), ctcp)==ctcp_methods.end()) {
      cout<<"Valid CTCP methods: "; for(auto& b: ctcp_methods) cout<<b<<" ";
      cout<<endl;
      cout<<"Using 'CF-CTCP'"<<endl;
      ctcp = "CF-CTCP";
    }
    cout<<"Dataset: "<<argv[1]<<endl;
    cout<<"Algorithm: "<<algorithm<<endl;
    cout<<"K: "<<argv[2]<<endl;
    cout<<"Branching: "<<branching<<endl;
    cout<<"Bounding: "<<bounding<<endl;
    cout<<"UBR2: "<<UBR2<<endl;
    cout<<"CTCP Method: "<<ctcp<<endl;
    cout<<"Heuristic: "<<heuristic<<endl;
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

CommandLine PARA;


