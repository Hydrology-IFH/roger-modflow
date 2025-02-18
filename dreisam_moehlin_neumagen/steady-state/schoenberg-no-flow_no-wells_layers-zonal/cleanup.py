from pathlib import Path
import os
import click
import glob

@click.option("-mr", "--model-run", type=int, default=0)
@click.command("main")
def main(model_run):

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.dis.grb"))
    for file in files:
        os.remove(file)

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.lst"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.sto"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.cbb"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.npf"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.ic"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.rcha"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.chd"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.dis"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.tdis"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.oc"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.nam"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.hds"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.ims"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.drn"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    base_path = Path(__file__).parent
    files = glob.glob(str(base_path / "output" / "dmn_run_*.wel"))
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    return


if __name__ == "__main__":
    main()
