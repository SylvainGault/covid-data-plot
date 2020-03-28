#!/usr/bin/env python3

import os
import contextlib
import subprocess
import argparse
import tempfile
import sklearn.preprocessing as skprep
import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize
import config
import db


def plot(cur, datasource, name, params={}):
    # Make the possibly missing directories
    if config.tmpdir:
        os.makedirs(config.tmpdir, exist_ok=True)
    os.makedirs(config.figdir, exist_ok=True)

    # Make the file path from the configured path
    gnuplotinc = os.path.join(config.gnuplotdir, "common.plt")
    figfile = os.path.join(config.figdir, "%s.png" % name)
    gnuplotfile = os.path.join(config.gnuplotdir, "%s.plt" % name)

    gnuplotcmd = ["gnuplot", "-d"]
    gnuplotcmd += ["-e", 'load "%s"' % gnuplotinc]
    gnuplotcmd += ["-e", 'set output "%s"' % figfile]

    with contextlib.ExitStack() as exitstack:
        for dataname, datasource in datasource.items():
            datafp = tempfile.NamedTemporaryFile("w+", dir=config.tmpdir)
            exitstack.enter_context(datafp)

            if cur is not None:
                it = cur.execute(datasource, params)
            else:
                it = datasource

            cnt = 0
            for row in it:
                rowstr = " ".join(str(f) for f in row)
                print(rowstr, file=datafp)
                cnt += 1

            print(cnt, "rows written")
            datafp.flush()
            gnuplotcmd += ["-e", '%s = "%s"' % (dataname, datafp.name)]

        for kv in params.items():
            gnuplotcmd += ["-e", '%s = %r' % kv]

        gnuplotcmd.append(gnuplotfile)
        subprocess.run(gnuplotcmd)




def plot_raw_data(cur):
    datasource = {
        'conf_time_fr': """
            SELECT date, confirmed
            FROM daily_update
            WHERE country='France'
            ORDER BY date
        """,
        'conf_time_ch': """
            SELECT date, confirmed
            FROM daily_update
            WHERE country='Chine'
            ORDER BY date
        """
    }
    plot(cur, datasource, "confirmed_time")

    datasource = {
        'conf_time_fr': """
            SELECT date, confirmed - lag(confirmed) OVER win
            FROM daily_update
            WHERE country='France'
            WINDOW win AS (ORDER BY date)
            ORDER BY date
        """,
        'conf_time_ch': """
            SELECT date, confirmed - lag(confirmed) OVER win
            FROM daily_update
            WHERE country='Chine'
            WINDOW win AS (ORDER BY date)
            ORDER BY date
        """
    }
    plot(cur, datasource, "diff_confirmed_time")

    datasource = {
        'conf_time_fr': """
            SELECT confirmed, confirmed - lag(confirmed) OVER win
            FROM daily_update
            WHERE country='France'
            WINDOW win AS (ORDER BY date)
            ORDER BY date
        """,
        'conf_time_ch': """
            SELECT confirmed, confirmed - lag(confirmed) OVER win
            FROM daily_update
            WHERE country='Chine'
            WINDOW win AS (ORDER BY date)
            ORDER BY date
        """
    }
    plot(cur, datasource, "diff_confirmed_confirmed")



def exp(X, a, b):
    return np.exp(a*X+b)



def sigma(X, a, b, c):
    return c / (1.0 + exp(X, -a, -b))



def get_dataframe(cnx, country):
    params = {'country': country}

    sql = """
        SELECT date, confirmed
        FROM daily_update
        WHERE country=:country
        ORDER BY date
    """
    return pd.read_sql_query(sql, cnx, params=params, parse_dates=["date"])



def fit_models(X, Y):
    scaler = skprep.StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1)

    poptexp, _ = sp.optimize.curve_fit(exp, X, Y)
    poptsig, _ = sp.optimize.curve_fit(sigma, X, Y, p0=[1.0, 1.0, Y.max()])

    return scaler, poptexp, poptsig



def dataframe_fit(cnx, country="France"):
    df = get_dataframe(cnx, country)
    X = df["date"].to_numpy().astype(np.float64)
    Y = df["confirmed"].to_numpy()

    scaler, poptexp, poptsig = fit_models(X, Y)
    X = scaler.transform(X.reshape(-1, 1)).reshape(-1)
    df["expmodel"] = exp(X, *poptexp)
    df["sigmoidmodel"] = sigma(X, *poptsig)

    return df, scaler, poptexp, poptsig



def extrapolate(days, df, scaler, poptexp, poptsig):
    day = np.timedelta64(1, "D")
    firstdate = df["date"].to_numpy().max() + day
    dates = np.arange(firstdate, firstdate + days * day, day)

    X = dates.astype(np.float64)
    X = scaler.transform(X.reshape(-1, 1)).reshape(-1)

    Yexp = exp(X, *poptexp)
    Ysig = sigma(X, *poptsig)

    columns = {
        "date": dates,
        "expmodel": Yexp,
        "sigmoidmodel": Ysig
    }
    newdf = pd.DataFrame(columns)

    return df.append(newdf, sort=False)



def plot_regression(cnx):
    fr = dataframe_fit(cnx, "France")
    ch = dataframe_fit(cnx, "Chine")

    dffr = extrapolate(30, *fr)
    dfch = extrapolate(30, *ch)

    datasource = {
        'conf_fit_fr': dffr.itertuples(index=False),
        'conf_fit_ch': dfch.itertuples(index=False)
    }

    plot(None, datasource, "confirmed_fit_time")



def main():
    parser = argparse.ArgumentParser(description="Plot the statistics")
    parser.add_argument("-f", "--figdir", help="Directory where to store the output figures (default: %s)" % config.figdir)
    parser.add_argument("-g", "--gnuplotdir", help="Directory where the gnuplot scripts are located (default: %s)" % config.gnuplotdir)
    parser.add_argument("-t", "--tmpdir", help="Directory where to store the temporary data files (default to system temporary directory)")

    args = parser.parse_args()

    if args.figdir is not None:
        config.figdir = args.figdir
    if args.gnuplotdir is not None:
        congnuplot.gnuplotdir = args.gnuplotdir
    if args.tmpdir is not None:
        config.tmpdir = args.tmpdir

    cnx = db.new_connection()
    cur = cnx.cursor()
    plot_raw_data(cur)
    plot_regression(cnx)

    cur.execute("PRAGMA optimize")



if __name__ == '__main__':
    main()
