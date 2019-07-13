import numpy as np
import csv
import sqlite3

from statistics import mean, median, stdev

conn = sqlite3.connect("results.db")
c = conn.cursor()

def insert():
    with open("results.csv") as f:
        reader = csv.reader(f, delimiter = ";")
        X = [row for row in reader]

    cur_ds, cur_s = None, None
    for row in X:
        if len(row) == 1:
            cur_ds, cur_s = row[0].split(" ")
        else:
            row += [cur_ds, cur_s]

    X = np.array([x for x in X if len(x) > 1 and x[0]!=''])

    c.execute("DROP TABLE IF EXISTS res")
    c.execute("""CREATE TABLE res (
        reward_fn text,
        bare real,
        svrc1 real,
        svrc100 real,
        gp0 real,
        gp1 real,
        gp2 real,
        ds text,
        scorer text
    )""")

    for x in X:
        c.execute("""INSERT INTO res VALUES(
            "{}", {}, {}, {}, {}, {}, {}, "{}", "{}"
        )""".format(*x))

    conn.commit()
    conn.close()

HEADER = [ "\\texttt{bare}", "\\texttt{SVR C1}"
         , "\\texttt{SVR C100}", "\\texttt{GP [1, 2]}"
         , "\\texttt{GP [1e-1, 1]}"
         , "\\texttt{GP [1e-3, 1]}" ]

TEXHEADER = """
\\begin{table}
\\begin{center}
  {\\scriptsize
  \\begin{tabu}{l|l|l|l|l|l|l}
"""

TEXFOOTER = """
  \\end{tabu} }
  \\caption{ %s }
\\end{center}
\\end{table}
"""

TEXHEADERLONG = """
\\begin{center}
{\\scriptsize
\\begin{longtabu}{l|l|l|l|l|l|l}
"""

TEXFOOTERLONG = """
\\caption{ %s }
\\end{longtabu} }
\\end{center}
"""

def res(X):
    res = {}
    for j in range(X.shape[1]):
        res[HEADER[j]] = [
            mean(X[:,j]), median(X[:,j]), stdev(X[:,j]),
        ]
    return res

def main():
    X = np.array([row for row in c.execute("""SELECT
        bare, svrc1, svrc100, gp0, gp1, gp2 FROM res""")])

    r = res(X)

    TEX = ["", "mean ", "median ", "$\\sigma$ "]
    for k, v in r.items():
        TEX[0] += "&{} ".format(k)
        for i, e in enumerate(v):
            TEX[i + 1] += "&{:.3f} ".format(e)
    TEX = [x + "\\\\" for x in TEX]
    TEX[0] += "\\hline"

    print(TEXHEADER)
    for t in TEX: print(t)
    print(TEXFOOTER % """Metrics over all data sets,
        scoring classifiers and reward functions.""")


    rew_fns = [row[0] for row in
        c.execute("""SELECT DISTINCT reward_fn FROM res""")
    ]

    TEXX = [""]

    for i, fn in enumerate(rew_fns):
        X = np.array([row for row in c.execute("""SELECT
            bare, svrc1, svrc100, gp0, gp1, gp2 FROM res
            WHERE reward_fn = \"{}\"""".format(fn)
        )])

        r = res(X)

        TEX = ["mean ", "median ", "$\\sigma$ "]
        for k, v in r.items():
            if i == 0: TEXX[0] += "&{} ".format(k)
            for j, e in enumerate(v):
                TEX[j] += "&{:.3f} ".format(e)

        TEXX += ["\\hline\n{} &&&&&& \\\\ &&&&&&".format(fn)]\
             + TEX

    TEXX = [x + "\\\\" for x in TEXX]
    print(TEXHEADERLONG)
    for t in TEXX: print(t)
    print(TEXFOOTERLONG % """Metrics for each reward
        function over all data sets and scoring
        classifiers.""")

    TEXX = [""]
    scorers = ["cp", "rf"]
    for i, scorer in enumerate(scorers):
        X = np.array([row for row in c.execute("""SELECT
            bare, svrc1, svrc100, gp0, gp1, gp2 FROM res
            WHERE scorer= \"{}\"""".format(scorer)
        )])

        r = res(X)

        TEX = ["mean ", "median ", "$\\sigma$ "]
        for k, v in r.items():
            if i == 0: TEXX[0] += "&{} ".format(k)
            for j, e in enumerate(v):
                TEX[j] += "&{:.3f} ".format(e)

        TEXX += ["\\hline\n{} &&&&&& \\\\ &&&&&&".format(scorer)]\
             + TEX

    TEXX = [x + "\\\\" for x in TEXX]
    print(TEXHEADER)
    for t in TEXX: print(t)
    print(TEXFOOTER % """Metrics for each scoring classifier
        over all data sets and reward functions.""")

    # magic
    X = np.array([row for row in c.execute("""SELECT
        bare, svrc1, svrc100, gp0, gp1, gp2 FROM res""")])

    # how often reg better than t
    count, amount = 0, 0.0
    for row in X:
        mx = max(row[1:])
        if mx > row[0]:
            count += 1
            amount += mx - row[0]

    print(count / len(X), amount / len(X))

if __name__ == "__main__":
    main()
    #insert()
