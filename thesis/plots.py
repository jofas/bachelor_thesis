import os
import csv

TRANS = {
    "bank"           : "\\texttt{bank}",
    "bank_additional" : "\\texttt{bank-additional}",
    "car"            : "\\texttt{car}",
    "credit_card"     : "\\texttt{credit card}",
    "usps"           : "\\texttt{usps}",
    "wine_quality"    : "\\texttt{wine}",
    "cp" : "\\texttt{cp}",
    "rf" : "\\texttt{rf}",
    "asymetric_1000"           : "\\texttt{asymmetric loss 1000}",
    "asymetric_200"            : "\\texttt{asymmetric loss 200}",
    "asymetric_50"             : "\\texttt{asymmetric loss 50}",
    "random"                   : "\\texttt{random loss}",
    "random_scaled_1_100"      : "\\texttt{random loss scaled 1:100}",
    "random_scaled_1_20"       : "\\texttt{random loss scaled 1:20}",
    "random_scaled_1_5"        : "\\texttt{random loss scaled 1:5}",
    "simple"                   : "\\texttt{simple}",
    "simple_scaled_1_100"      : "\\texttt{simple scaled 1:100}",
    "simple_scaled_1_20"       : "\\texttt{simple scaled 1:20}",
    "simple_scaled_1_5"        : "\\texttt{simple scaled 1:5}",
    "asymetric_1000_abstain"   : "\\texttt{asymmetric 1000 abstain}",
    "asymetric_200_abstain"    : "\\texttt{asymmetric 200 abstain}",
    "asymetric_50_abstain"     : "\\texttt{asymmetric 50 abstain}",
    "random_gain"              : "\\texttt{random}",
    "random_gain_scaled_1_100" : "\\texttt{random scaled 1:100}",
    "random_gain_scaled_1_20"  : "\\texttt{random scaled 1:20}",
    "random_gain_scaled_1_5"   : "\\texttt{random scaled 1:5}",
    "GP [1,2]"     : "\\texttt{GP [1, 2]}",
    "GP [1e-1,1]"  : "\\texttt{GP [1e-1, 1]}",
    "GP [1e-3, 1]" : "\\texttt{GP [1e-3, 1]}",
    "SVR RBF C100" : "\\texttt{SVR C100}",
    "SVR RBF C1"   : "\\texttt{SVR C1}",
    "bare"         : "\\texttt{bare}"
}

ROOT = "../experiments/"

def rename():
    for dir in os.listdir(ROOT):
        for scorer in os.listdir(ROOT + dir):
            for rew_fn in os.listdir(
                ROOT + dir + "/" + scorer
            ):
                for ds in os.listdir(
                    ROOT + dir + "/" + scorer + "/"+rew_fn
                ):
                    path = ROOT+dir+"/"+scorer+"/"+rew_fn
                    file = path + "/" + ds
                    file_new = path + "/" \
                             + ds.replace(" ", "")

                    if ds == "result.csv": continue

                    #if file == file_new:
                    #    print(file)
                    #    os.remove(file_new)
                    #else:
                    os.rename(file, file_new)

                    with open(file_new) as f:
                        reader = \
                            csv.reader(f, delimiter=";")
                        X = [row for row in reader]
                    with open(file_new, "w") as f:
                        for row in X:
                            w = ""
                            for x in row:
                                w += "{:.6f}, ".format(float(x))
                            f.write("{}\n".format(w[:-2]))

FILES = [ "Trainingset.csv", "Testset.csv", "SVRRBFC1.csv"
        , "SVRRBFC100.csv", "GP[1,2].csv", "GP[1e-1,1].csv"
        , "GP[1e-3,1].csv" ]

TEMPLATE = """
  \\begin{figure}[H]
  \\begin{center}
  \\begin{tikzpicture}[scale=2]
    \\datavisualization[
      scientific axes=clean,
      visualize as line/.list={trainingset, testset, svrc1,
                               svrc100, gp0, gp1, gp2},
      style sheet=strong colors,
      trainingset={label in legend={text=Training set}},
      testset={label in legend={text=Test set}},
      svrc1={label in legend={text=\\texttt{SVR C1}}},
      svrc100={label in legend={text=\\texttt{SVR C100}}},
      gp0={label in legend={text=\\texttt{GP [1, 2]}}},
      gp1={label in legend={text=\\texttt{GP [1e-1, 1]}}},
      gp2={label in legend={text=\\texttt{GP [1e-3, 1]}}},
      x axis = {label=score},
      y axis = {label=normalized reward},
    ]
    data[
      headline={x, y},
      set=trainingset,
      read from file={
        %s
      }
    ]
    data[
      headline={x, y},
      set=testset,
      read from file={
        %s
      }
    ]
    data[
      headline={x, y},
      set=svrc1,
      read from file={
        %s
      }
    ]
    data[
      headline={x, y},
      set=svrc100,
      read from file={
        %s
      }
    ]
    data[
      headline={x, y},
      set=gp0,
      read from file={
        %s
      }
    ]
    data[
      headline={x, y},
      set=gp1,
      read from file={
        %s
      }
    ]
    data[
      headline={x, y},
      set=gp2,
      read from file={
        %s
      }
    ]
    ;
  \\end{tikzpicture}
  \\end{center}

  \\caption{Results from the %s reward function.}
  \\end{figure}
"""

def main():
    TEX = {}
    for dir in os.listdir(ROOT):

        t = dir[:-3]
        if t not in TEX:
            TEX[t] = ""#"\\subsection{%s}" % TRANS[t]

        for scorer in os.listdir(ROOT + dir):
            #TEX[t] += "\n\n\\texttt{%s}\n\n" % scorer

            for rew_fn in os.listdir(
                ROOT + dir + "/" + scorer
            ):

                path = ROOT+dir+"/"+scorer+"/"+rew_fn
                fs = [path + "/" + file for file in FILES]
                fs += [TRANS[rew_fn]]
                TEX[t] += TEMPLATE % tuple(fs)

            break

    keys = sorted([t for t in TEX])
    print(TEX["usps"])
    #for k in keys:
    #print(k)
    #print(TEX[k])

if __name__ == "__main__":
    main()
    #rename()
