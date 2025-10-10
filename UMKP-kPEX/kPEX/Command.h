
class CommandLine
{
public:

	string branching, bounding, alg;
	bool UBR2;
	bool BR1, BR2, RR1, RR2, RR3;



  CommandLine(int _argc, char **_argv) : argc(_argc), argv(_argv)
  {
    alg = GetOptionValue("-algorithm", "UMKP");
    branching = GetOptionValue("-branching", "Default-Br");
    bounding = GetOptionValue("-bounding", "None");
    UBR2 = GetOptionValue("-UBR2", "true") == "true";
    BR1 = GetOptionValue("-BR1", "true") == "true";
    BR2 = GetOptionValue("-BR2", "true") == "true";
    RR1 = GetOptionValue("-RR1", "true") == "true";
    RR2 = GetOptionValue("-RR2", "true") == "true";
    RR3 = GetOptionValue("-RR3", "true") == "true";

    cout<<"----------------------Parameters-----------------------\n";
    vector<string> branchings={"Default-Br", "Pivot-Br", "S-Br", "R-Br", "SR-Br"};
    if(find(branchings.begin(), branchings.end(), branching)==branchings.end()) {
      cout<<"Not a valid selection for branching, using Default-Br"<<endl;
      cout<<"Valid branching methods: "; for(auto& b: branchings) cout<<b<<" ";
      cout<<endl;
      branching = "Default-Br";
    }

    vector<string> boundings={"None", "S-Bound", "R-Bound", "SR-Bound", "AltRB"};
    if(find(boundings.begin(), boundings.end(), bounding)==boundings.end()) {
      cout<<"Not a valid selection for bounding, using 'None'"<<endl;
      cout<<"Valid bounding methods: "; for(auto& b: boundings) cout<<b<<" ";
      cout<<endl;
      branching = "None";
    }


    cout<<"Dataset: "<<argv[1]<<endl;
    cout<<"Algorithm: "<<alg<<endl;
    cout<<"K: "<<argv[2]<<endl;
    cout<<"Branching: "<<branching<<endl;
    cout<<"Bounding: "<<bounding<<endl;
    cout<<"UBR2: "<<UBR2<<endl;
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
