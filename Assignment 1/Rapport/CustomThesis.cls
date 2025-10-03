\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CustomThesis}[24/09/25 A Thesis template for university reports]

% --- Base Class Setup ---
\LoadClass[]{article} % Base class
\RequirePackage{iftex}    % Detect LaTeX engine
\RequirePackage[a4paper, total={6.5in, 9in}]{geometry}

\setlength{\parindent}{0pt} % Set the paragraph indent

\RequirePackage{fancyhdr} % To make fancy headers
\pagestyle{fancy}

\RequirePackage{xcolor}
\PassOptionsToPackage{table,xcdraw}{xcolor}

\newcommand{\printtitle}{\noindent\textcolor{StyleColorDark}{\Huge \StyleFontBold \@title}}
\newcommand{\printdepartment}{\noindent\textcolor{StyleColorDark}{\StyleFontRegular \@department}}
\newcommand{\printshortdesc}{\noindent\textcolor{StyleColorDark}{\huge\StyleFontRegular \@shortdesc}}

\newcommand{\department}[1]{\gdef\@department{#1}}
\newcommand{\shortdesc}[1]{\gdef\@shortdesc{#1}}

% \RequirePackage{moreverb,savetrees}
% % Compile with  --enable-write18 or --shell-escape options   
% \immediate\write18{texcount -tex -sum \jobname.tex > /tmp/wordcount.tex}

% Defaults so they don’t cause errors if not set
\gdef\@department{}
\gdef\@shortdesc{}

% --- Class Options ---
\RequirePackage{kvoptions}
\RequirePackage{ifthen}
\SetupKeyvalOptions{
  family=CustomThesis,
  prefix=CustomThesis@
}
\DeclareStringOption[default]{style} % Determines which style to use, defaults to default
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{article}} % Passes through the normal class options
\ProcessKeyvalOptions*

% Define a boolean for the style
\newif\ifAUstyle
\AUstylefalse % default

% Set it when the style option is chosen
\DeclareOption{AU}{\AUstyletrue}
\ProcessOptions\relax

% --- Base Colors ---
\definecolor{DarkGray}{gray}{0.67}
\definecolor{Gray}{RGB}{191, 191, 191}
\definecolor{LightGray}{RGB}{248, 248, 248}

% --- Styles ---
% Default style
\ifthenelse{\equal{\CustomThesis@style}{default}}{%
    % Default Colors
    % Gray:
    \definecolor{StyleColorDark}{RGB}{0, 0, 0}
    \definecolor{StyleColor}{RGB}{40, 40, 40}
    \definecolor{StyleColorLight}{RGB}{245, 245, 245}
    % Green:
    % \definecolor{StyleColorDark}{RGB}{0, 60, 30}
    % \definecolor{StyleColor}{RGB}{0, 120, 60}
    % \definecolor{StyleColorLight}{RGB}{220, 240, 220}
    % Orange:
    % \definecolor{StyleColorDark}{RGB}{160, 60, 10}
    % \definecolor{StyleColor}{RGB}{225, 110, 50}
    % \definecolor{StyleColorLight}{RGB}{255, 240, 230}
    % Purple:
    % \definecolor{StyleColorDark}{RGB}{60, 0, 80}
    % \definecolor{StyleColor}{RGB}{120, 60, 140}
    % \definecolor{StyleColorLight}{RGB}{235, 230, 245}

    % Default Fonts
    \RequirePackage[T1]{fontenc}
    \RequirePackage[utf8]{inputenc}
    \RequirePackage{lmodern}
    \RequirePackage[scaled]{helvet}
    \RequirePackage[scaled]{beramono} % Bera Mono font, close to JetBrains Mono
    % \RequirePackage[scaled]{DejaVuSansMono}
    % \RequirePackage[scaled]{FiraMono} % Maybe not portable


    \newcommand{\StyleFontBold}{\sffamily\bfseries}
    \newcommand{\StyleFontRegular}{\sffamily}
    \newcommand{\StyleFontItalic}{\sffamily\itshape}

    % Default front page
    \newcommand{\FrontPageLogo}{}

    % Default Header
    \fancyhead[R]{\color{StyleColorDark}\StyleFontRegular \@title}
}{}

% Aarhus University style
\ifthenelse{\equal{\CustomThesis@style}{AU}}{%
    % AU Colors
    \definecolor{StyleColorDark}{RGB}{0, 37, 70}
    \definecolor{StyleColor}{RGB}{0, 61, 115}
    \definecolor{StyleColorLight}{RGB}{231, 238, 254}

    % AU Fonts
    \ifLuaTeX
        % --- LuaLaTeX with style specific fonts ---
        \RequirePackage{fontspec}
        \newfontface\StyleFontBold{AUPassataRegularBold}[
            Path=./fonts/,
            Extension=.ttf,
        ]
        \newfontface\StyleFontRegular{AUPassataRegular}[
            Path=./fonts/,
            Extension=.ttf,
        ]
        \newfontface\StyleFontItalic{AUPassataRegularOblique}[
            Path=./fonts/,
            Extension=.ttf,
        ]
    \else
        % --- Fallback for pdfLaTeX, XeLaTeX, latexmk, etc. ---
        \RequirePackage[T1]{fontenc}
        \RequirePackage[utf8]{inputenc}
        \RequirePackage{lmodern}
        \RequirePackage[scaled]{helvet}
        \RequirePackage[scaled]{beramono} % Bera Mono font, close to JetBrains Mono

        % Define AU font replacements
        \newcommand{\StyleFontBold}{\sffamily\bfseries}
        \newcommand{\StyleFontRegular}{\sffamily}
        \newcommand{\StyleFontItalic}{\sffamily\itshape}

        % Print out "warning" that ideal font isn't used
        \PackageWarning{CustomThesis}{Using fallback fonts, switch to LuaLaTeX to get style specific fonts}
    \fi

    % AU front page
    \newcommand{\FrontPageLogo}{%
        \begin{figure}[h]
            \includegraphics[width=0.6\textwidth]{Figure/misc/aulogo_dk_var1_blaa.eps}
        \end{figure}
    }

    % AU Header
    \fancyhead[R]{\includegraphics[height=16px]{Figure/misc/aulogo_dk_var2_blaa.eps}}
}{}

% --- Captions ---
\RequirePackage{caption}
\DeclareCaptionFont{StypeCaption}{\color{StyleColor}\StyleFontRegular}
\DeclareCaptionFont{StypeCaptionBold}{\color{StyleColorDark}\StyleFontBold}
\captionsetup{font=StypeCaption, labelfont=StypeCaptionBold}

% --- Section Titles ---
\RequirePackage{titlesec}
\titleformat{\section}
{\color{StyleColorDark}\StyleFontBold\Large}
{\color{StyleColorDark}\thesection}{1em}{}

\titleformat{\subsection}
{\color{StyleColor}\StyleFontBold\large}
{\color{StyleColor}\thesubsection}{1em}{}

\titleformat{\subsubsection}
{\color{StyleColor}\StyleFontBold\normalsize}
{\color{StyleColor}\thesubsubsection}{1em}{}

% --- Headers and Footers ---
\fancyhead[L]{\color{StyleColorDark}\StyleFontRegular\nouppercase{\leftmark}}
\renewcommand{\headrule}{{\color{StyleColorDark}\hrule width\headwidth height\headrulewidth}}
\setlength{\headheight}{24pt}
\fancyfoot[L]{}
\fancyfoot[C]{}
\fancyfoot[R]{\color{StyleColorDark}\StyleFontBold\thepage}

\let\oldps@plain\ps@plain
\let\ps@plain\ps@fancy

% --- TOC/LOF/LOT formatting ---
% Formatting for Table of Contents
\RequirePackage{tocloft}
\renewcommand{\cfttoctitlefont}{\color{StyleColorDark}\StyleFontBold\Large}
\renewcommand{\cftsecfont}{\color{StyleColorDark}\StyleFontBold}
\renewcommand{\cftsubsecfont}{\color{StyleColorDark}\StyleFontRegular}
\renewcommand{\cftsubsubsecfont}{\color{StyleColorDark}\StyleFontRegular}
\renewcommand{\cftsecpagefont}{\color{StyleColorDark}\StyleFontBold}
\renewcommand{\cftsubsecpagefont}{\color{StyleColorDark}\StyleFontRegular}
\renewcommand{\cftsubsubsecpagefont}{\color{StyleColorDark}\StyleFontRegular}

% Same for List of Figures
\renewcommand{\cftloftitlefont}{\color{StyleColorDark}\StyleFontBold\Large}
\renewcommand{\cftfigfont}{\color{StyleColorDark}\StyleFontRegular}
\renewcommand{\cftfigpagefont}{\color{StyleColorDark}\StyleFontRegular}

% Same for List of Tables
\renewcommand{\cftlottitlefont}{\color{StyleColorDark}\StyleFontBold\Large}
\renewcommand{\cfttabfont}{\color{StyleColorDark}\StyleFontRegular}
\renewcommand{\cfttabpagefont}{\color{StyleColorDark}\StyleFontRegular}

% --- Hyperlinks ---
\RequirePackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=StyleColorDark,
    citecolor=StyleColorDark,
    urlcolor=black,
    breaklinks=true
}

% --- Counters ---
% Section based numbers for figures etc.
\counterwithin{equation}{section}
\counterwithin{figure}{section}
\counterwithin{table}{section}

% Add for code blocks:
% \RequirePackage{float}
% \newfloat{code}{htbp}{lop}
% \floatname{code}{Code Block}
% \counterwithin{code}{section}

% Load preamble
\RequirePackage{CustomThesis}