//---------------------------------------------------------------------------
//
// Name:        icpdApp.cpp
// Author:      Adriel Mota Ziesemer Junior
// Created:     20/9/2007 19:13:08
// Description: 
//
//---------------------------------------------------------------------------

#include "icpdApp.h"

DECLARE_APP(icpdFrmApp)
IMPLEMENT_APP_NO_MAIN(icpdFrmApp);

bool icpdFrmApp::Initialize(int& argc, wchar_t **argv) { 

    static const wxCmdLineEntryDesc desc[] = { 
        { wxCMD_LINE_SWITCH, wxT_2("s"), wxT_2("shell"), wxT_2("Run in shell mode") }, 
        // { wxCMD_LINE_SWITCH, wxT_2("c"), wxT_2("commands"), wxT_2("Show all available commands") },
        { wxCMD_LINE_PARAM, NULL, NULL, wxT_2("FILENAME"), wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL },
        { wxCMD_LINE_NONE }
    }; 

    wxCmdLineParser parser(desc, argc, argv); 
    if (parser.Parse(true) != 0) { 
        exit(1); 
    } 
    
    gui_enabled = !parser.Found(wxT_2("shell"));

    // cout << " STAT:" << parser.Parse(true) << endl;
    // cout << " FILE:" << parser.GetParam(0).mb_str() << endl;
    // cout << " CONT:" << parser.GetParamCount() << endl;

    if (parser.GetParamCount() == 1) {
        cmdFilename = parser.GetParam(0);
    }

    if (gui_enabled) { 
        return wxApp::Initialize(argc, argv); 
    } else { 
        return wxAppConsole::Initialize(argc, argv); 
    }
} 

bool icpdFrmApp::OnInit()
{
    // Shell mode is handled entirely in main() below; this path is GUI-only.
    setlocale(LC_ALL, "C");
    wxInitAllImageHandlers();
    IcpdFrm *frame = new IcpdFrm(NULL);
    SetTopWindow(frame);
    frame->Show();
    return true;
}

int icpdFrmApp::OnExit(){
    return EXIT_SUCCESS;
}

void icpdFrmApp::CleanUp()
{ 
    if (gui_enabled) { 
        wxApp::CleanUp(); 
    } else { 
        wxAppConsole::CleanUp(); 
    } 
} 

HybridTraits *icpdFrmApp::CreateTraits()
{ 
    return new HybridTraits(gui_enabled); 
} 

// ── Custom main: shell mode bypasses wxEntry/Cocoa entirely ──────────────────
int main(int argc, char **argv)
{
    // Quick scan for --shell / -s flag and optional script filename.
    // Done here before any wx/Cocoa init so stdin/stdout stay connected.
    bool shell_mode = false;
    const char *script_file = nullptr;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--shell" || arg == "-s")
            shell_mode = true;
        else if (!arg.empty() && arg[0] != '-' && shell_mode && !script_file)
            script_file = argv[i];
    }

    if (!shell_mode) {
        // GUI mode: normal wx/Cocoa path.
        wxDISABLE_DEBUG_SUPPORT();
        return wxEntry(argc, argv);
    }

    // ── Shell mode: pure C++, no wx event loop, no Cocoa ─────────────────────
    setlocale(LC_ALL, "C");
    DesignMng designmng;
    std::string cmd;

    // Load astran.cfg
    const char *env_path = getenv("ASTRAN_PATH");
    std::string astran_cfg = env_path
        ? (std::string(env_path) + "/bin/astran.cfg")
        : "astran.cfg";
    {
        std::ifstream afile(astran_cfg);
        if (afile) {
            afile.close();
            cmd = "read " + astran_cfg;
            std::cout << "astran> " << cmd << "\n";
            designmng.readCommand(cmd);
        }
    }

    if (script_file) {
        // Batch mode: read script and exit.
        std::ifstream ifile(script_file);
        if (!ifile) {
            std::cerr << "ERROR: File '" << script_file << "' doesn't exist\n";
            return 1;
        }
        ifile.close();
        cmd = std::string("read ") + script_file;
        std::cout << "astran> " << cmd << "\n";
        designmng.readCommand(cmd);
        return 0;
    }

    // Interactive REPL
    std::cout << "Astran shell ready. Type HELP for commands, EXIT to quit.\n" << std::flush;
    while (true) {
        std::cout << "astran> " << std::flush;
        if (!std::getline(std::cin, cmd)) break;
        designmng.readCommand(cmd);
    }
    return 0;
}
