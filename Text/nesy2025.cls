% Copied from clear2024.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nesy2025}[2025/05/09 v1.1 NeSy 2025 Proceedings]
\newcommand{\todo}{\color{red}}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{vol 284}
\jmlryear{2025}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\ifanonsubmission
 \newcommand{\clearauthor}[1]{}
 \author{Author names withheld}
 \editor{Under Review for NeSy 2025}
 \jmlrworkshop{19th Conference on Neurosymbolic Learning and Reasoning}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\clearauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{Leilani H. Gilpin, Eleonora Giunchiglia, Pascal Hitzler, and Emile van Krieken}
  \jmlrworkshop{19th Conference on Neurosymbolic Learning and Reasoning}
 \fi
\fi


