#include "app.h"

#include <wx/wx.h>
#include <wx/msgdlg.h>

#include <cstring>

#include <string>
using std::string;

BEGIN_EVENT_TABLE(wxSightApp, wxApp)
	EVT_MENU(wxID_NEW,wxSightApp::OnNew)
END_EVENT_TABLE()

// -----------------------------------------------------------------------------

char* narrow( const wxString& str ) {
	const wxScopedCharBuffer utf8 = str.utf8_str();
	const char* src = utf8.data() ? utf8.data() : "";
	char* c = new char[std::strlen(src) + 1];
	std::strcpy(c, src);
	return c;
}

bool wxSightApp::OnInit() {
	char ** args = new char*[argc];
	for ( int i = 0; i < argc; i++ )
		args[i] = narrow( argv[i] );
	
    glutInit( &argc, args );
	#ifdef __WXMAC__
		// wxApp::SetExitOnFrameDelete(false);
		wxMenuBar *menuBar = new wxMenuBar;
		
		wxMenu *menuFile = new wxMenu;
		menuFile->Append( wxID_NEW, wxString(_("New\tCtrl+N") ) );		
		menuBar->Append( menuFile, wxString(_("&File")));

		wxMenuBar::MacSetCommonMenuBar(menuBar);
	#endif

	wxInitAllImageHandlers();

	if ( argc == 2 )
		loadFile( argv[1] );
#ifndef	__WXMAC__
	else
		loadFile( _("") );
#endif	
	
  return true;
}  // end event

// -----------------------------------------------------------------------------

void wxSightApp::loadFile( const wxString &fileName ) {
	if ( clsFrames.size() > 0 ) {
		//wxBell();
		wxMessageBox( _("Currentely SightGL can manipulate just one window at time."), _("Information"), wxOK|wxICON_INFORMATION );
	} // end if
	else {
		wxSightFrame *newFrame = new wxSightFrame( _T("SightGL"), wxDefaultPosition, wxSize(800,600) );
		newFrame->Show(TRUE);
		newFrame->Initialize();

		clsFrames.push_back( newFrame );
		if ( !fileName.IsEmpty() )
			newFrame->openFile( fileName.ToAscii() );
		
	} // end else
} // end method

// -----------------------------------------------------------------------------

void wxSightApp::MacOpenFile(const wxString &fileName) {
	loadFile( fileName );
} // end method

// -----------------------------------------------------------------------------

void wxSightApp::OnNew(wxCommandEvent& event) {
	MacOpenFile( wxString(_("")) );
} // end method
