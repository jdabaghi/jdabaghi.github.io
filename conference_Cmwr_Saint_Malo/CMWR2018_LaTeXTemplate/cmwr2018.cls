\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cmwr2018}[2017/09/01 CMWR2018 abstract class]

\DeclareOption{12pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\LoadClass[10pt,a4paper]{article}

\RequirePackage[lmargin=2cm,rmargin=2cm,tmargin=1.5cm,bmargin=2cm]{geometry}
\RequirePackage[small,compact]{titlesec}
\RequirePackage[small,it]{caption}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{url}
\RequirePackage{color}

\setlength{\abovecaptionskip}{6pt}

\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@makecmwrtitle
      \else
        \twocolumn[\@makecmwrtitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@makecmwrtitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@makecmwrtitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
\thispagestyle{empty}
 \noindent\textbf{Key words:\space}\@keywords
 \par
}
\def\@makecmwrtitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
  \end{center}%
  \par
  \vskip 1.5em}

\endinput
