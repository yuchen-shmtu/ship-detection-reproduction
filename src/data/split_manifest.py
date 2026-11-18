"""Recording-level split manifest of the datasets: splits/record_level_split.json.

A recording is the unit that is never divided between train, val and test:

  session   SeaShips capture session: one camera's frames in time order (camera and time read from the
            on-screen display, seaships_osd.py), cut at gaps of more than 600 s (build_seaships_splits.py)
  smdplus   SMD-Plus video (the authors' video-level split; five train videos held out as val)
  aboships  ABOships video; whole days go to one subset, so all videos of a day share it
  wsodd     WSODD capture session, as for SeaShips (build_wsodd.py)
  smd       Singapore Maritime Dataset video (zero-shot test only)

McShips (web photos) and the SeaShips random split (frames drawn at random, kept for the leak study)
have no recordings and are listed by image. The recording-level bootstrap of the evaluation
(src/evaluation/compare_runs.py --cluster, boot_cache.py) resamples the same units: sessions and
videos (ABOships: the videos of its two test days), and McShips photos.

  check  compare data/<dataset>/splits/ (written by the build scripts) and the recording tables
         (groups.csv, frames.csv) with the manifest; exit status 1 on any difference
  build  write the manifest from data/; fails if a recording spans two subsets

Image ids are the stems the build scripts write: data/<dataset>/images/<id>.jpg.

    python src/data/split_manifest.py check
    python src/data/split_manifest.py build
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST = ROOT / "splits" / "record_level_split.json"
SUBSETS = ("train", "val", "test")

# key -> (data folder, split-file prefix, recording, how recordings are assigned to subsets, source)
INFO = {
    "session": ("seaships", "session", "capture session: one camera's frames in time order, cut at gaps of more "
                "than 600 s (camera and time read from the on-screen display)",
                "whole sessions per subset; sizes as the random protocol, class proportions matched, passenger "
                "ship in every subset",
                "SeaShips (Shao et al., IEEE Trans. Multimedia 2018), public 7,000-image subset as exported by "
                "Roboflow (Kaggle nam97595/seaships7k, 6,979 images)"),
    "random": ("seaships", "random", None,
               "frame-level random split for comparison with published results: the uploader's test set, val drawn "
               "at random from its train set; adjacent frames of one session fall on both sides",
               "as session"),
    "smdplus": ("smdplus", "smdplus", "video",
                "the authors' video-level train / test split; five whole train videos form val; every 5th annotated "
                "frame",
                "SMD-Plus (Kim et al., J. Mar. Sci. Eng. 2022, 10, 377), relabelled VIS videos of the Singapore "
                "Maritime Dataset"),
    "aboships": ("aboships", "aboships", "video (one trip; a frame every 15 s)",
                 "whole days per subset: test 2018-06-26 and 06-29, val 06-28 and 07-06, train the other nine days",
                 "ABOships (Iancu et al., Remote Sens. 2021), Zenodo record 4736931, the 9,041 labelled frames"),
    "mcships": ("mcships", "mcships", None,
                "the official ImageSets split; test_dedup leaves out 116 test photos with a near-copy in train "
                "(32x32 thumbnail cosine >= 0.98)",
                "McShips lite (Zheng & Zhang, ICME 2020)"),
    "wsodd": ("wsodd", "wsodd", "capture session: one camera's frames in time order, cut at gaps of more than "
              "600 s; the untimed phone photos form one group",
              "whole sessions per subset, about 70 / 10 / 20% of the frames, chosen for per-class box shares near "
              "those targets with every class in train and test",
              "WSODD (Zhou et al., Front. Neurorobot. 2021)"),
    "smd": ("smd_vis", None, "video", "test only: every 5th annotated frame of the 40 annotated VIS videos, for "
            "zero-shot tests of SeaShips-trained detectors",
            "Singapore Maritime Dataset (Prasad et al., IEEE T-ITS 2017), Kaggle mmichelli/singapore-maritime-dataset"),
}


def read_split(folder, name):
    f = DATA / folder / "splits" / (name + ".txt")
    return f.read_text().split() if f.exists() else None


def subsets_of(folder, prefix):
    """image -> subset from the split lists, or None if they are not built."""
    lists = {s: read_split(folder, "%s_%s" % (prefix, s)) for s in SUBSETS}
    if any(v is None for v in lists.values()):
        return None
    out = {}
    for s, ims in lists.items():
        for im in ims:
            if im in out:
                sys.exit("%s: %s is in %s and %s" % (prefix, im, out[im], s))
            out[im] = s
    return out


def recording_table(key):
    """image -> (recording, extra fields of that recording) from the dataset's recording table."""
    folder = INFO[key][0]
    if key == "session":
        g = pd.read_csv(DATA / folder / "groups.csv", dtype=str, keep_default_na=False)
        ts = g[g.timestamp != ""].groupby("session").timestamp
        span = dict(zip(ts.min().index, zip(ts.min(), ts.max())))
        cam = g.groupby("session").camera.first().to_dict()
        site = g.groupby("session").site.first().to_dict()
        return {r.id: (r.session, dict(camera=cam[r.session], site=site[r.session],
                                       start=span.get(r.session, (None, None))[0],
                                       end=span.get(r.session, (None, None))[1])) for r in g.itertuples()}
    f = pd.read_csv(DATA / folder / "frames.csv", dtype=str, keep_default_na=False)
    if key == "wsodd":
        ts = f[f.time != ""].groupby("video").time
        span = dict(zip(ts.min().index, zip(ts.min(), ts.max())))
        src = f.groupby("video").source.first().to_dict()
        return {r.image: (r.video, dict(source=src[r.video], start=span.get(r.video, (None, None))[0],
                                        end=span.get(r.video, (None, None))[1])) for r in f.itertuples()}
    if key == "aboships":
        return {r.image: (r.video, dict(day="%s-%s-%s" % (r.date[:4], r.date[4:6], r.date[6:]))) for r in f.itertuples()}
    return {r.image: (r.video, dict(platform=r.subset)) for r in f.itertuples()}  # smdplus, smd: onshore / onboard


def class_names(key):
    folder, prefix = INFO[key][:2]
    y = DATA / folder / "yolo" / ("%s.yaml" % ("session" if key == "random" else prefix or "all"))
    names = yaml.safe_load(y.read_text())["names"]
    return [names[k] for k in sorted(names)]


def recordings(table, subset_of):
    """[{id, subset, <fields>, n_images, images}], sorted by subset and id; fails if one spans two subsets."""
    members = {}
    for im in sorted(subset_of):
        members.setdefault(table[im][0], []).append(im)
    out = []
    for rec, ims in members.items():
        s = {subset_of[im] for im in ims}
        if len(s) != 1:
            sys.exit("recording %s spans %s" % (rec, sorted(s)))
        out.append(dict(id=rec, subset=s.pop(), **table[ims[0]][1], n_images=len(ims), images=ims))
    return sorted(out, key=lambda d: (SUBSETS.index(d["subset"]), d["id"]))


def counts(subset_of, recs=None):
    c = {s: dict(images=sum(v == s for v in subset_of.values())) for s in SUBSETS}
    for s in SUBSETS:
        if recs is not None:
            c[s]["recordings"] = sum(r["subset"] == s for r in recs)
        if not c[s]["images"]:
            del c[s]
    return c


def build():
    datasets = {}
    for key, (folder, prefix, rec, rule, source) in INFO.items():
        if prefix is None:  # smd: every frame in frames.csv is test
            table = recording_table(key)
            subset_of = {im: "test" for im in table}
        else:
            subset_of = subsets_of(folder, prefix)
            if subset_of is None:
                sys.exit("data/%s/splits/%s_*.txt missing: build the dataset first" % (folder, prefix))
            table = recording_table(key) if rec else None
        e = dict(dataset=folder if folder != "smd_vis" else "smd", recording=rec, split_rule=rule, source=source,
                 classes=class_names(key))
        if rec:
            missing = sorted(set(subset_of) - set(table))
            if missing:
                sys.exit("%s: %d images without a recording, e.g. %s" % (key, len(missing), missing[0]))
            e["recordings"] = recordings(table, subset_of)
            e["counts"] = counts(subset_of, e["recordings"])
            if key == "aboships":
                days = {}
                for r in e["recordings"]:
                    days.setdefault(r["day"], set()).add(r["subset"])
                if any(len(v) > 1 for v in days.values()):
                    sys.exit("aboships: a day spans two subsets")
                e["days"] = {d: v.pop() for d, v in sorted(days.items())}
                for s in SUBSETS:
                    e["counts"][s]["days"] = sum(v == s for v in e["days"].values())
        else:
            e["counts"] = counts(subset_of)
            e["images"] = {s: sorted(im for im, v in subset_of.items() if v == s) for s in SUBSETS}
        if key == "mcships":
            e["test_dedup"] = read_split(folder, "mcships_testdedup")
            e["counts"]["test_dedup"] = dict(images=len(e["test_dedup"]))
        datasets[key] = e
    session_test = {im for r in datasets["session"]["recordings"] if r["subset"] == "test" for im in r["images"]}
    cross = sorted(session_test & set(datasets["random"]["images"]["test"]))
    datasets["random"]["cross_test"] = cross
    datasets["random"]["counts"]["cross_test"] = dict(images=len(cross))
    m = dict(
        name="Recording-level splits of the ship detection datasets",
        version=1,
        notes=["Image ids are the file stems written by the build scripts in src/data: data/<dataset>/images/<id>.jpg.",
               "Keys are the protocol names of the code (--protocol); 'smd' is the zero-shot test set.",
               "A recording is the unit kept in one subset; the recording-level bootstrap resamples the same units "
               "(McShips: photos).",
               "random.cross_test = random test & session test: SeaShips test images that neither protocol trained "
               "on, where the leak of the random split is measured.",
               "Check a rebuilt data/ against this file: python src/data/split_manifest.py check"],
        datasets=datasets)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(dumps(m) + "\n", encoding="utf-8")
    for key, e in datasets.items():
        print("%-9s %s" % (key, json.dumps(e["counts"])))
    print("wrote", MANIFEST)


def dumps(o, ind=0):
    """JSON, indented, with each recording and each image list on one line."""
    sp = "  " * ind
    if isinstance(o, dict):
        return "{\n%s\n%s}" % (",\n".join("%s  %s: %s" % (sp, json.dumps(k), dumps(v, ind + 1)) for k, v in o.items()),
                               sp) if o else "{}"
    if isinstance(o, list) and o and isinstance(o[0], dict):
        return "[\n%s\n%s]" % (",\n".join("%s  %s" % (sp, json.dumps(v, ensure_ascii=False)) for v in o), sp)
    return json.dumps(o, ensure_ascii=False)


def check():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for key, e in m["datasets"].items():
        folder, prefix, rec = INFO[key][:3]
        if rec:
            want = {}
            for r in e["recordings"]:
                for im in r["images"]:
                    if im in want:
                        print("%-9s manifest lists %s twice" % (key, im))
                        bad += 1
                    want[im] = (r["subset"], r["id"])
        else:
            want = {im: (s, None) for s, ims in e["images"].items() for im in ims}
        if not (DATA / folder).exists():
            print("%-9s data/%s/ not built: skipped" % (key, folder))
            continue
        if prefix is None:
            have = {im: "test" for im in pd.read_csv(DATA / folder / "frames.csv", dtype=str).image}
        else:
            have = subsets_of(folder, prefix)
            if have is None:
                print("%-9s data/%s/splits/%s_*.txt missing: skipped" % (key, folder, prefix))
                continue
        diff = [im for im in set(want) | set(have) if want.get(im, (None,))[0] != have.get(im)]
        if rec:
            table = recording_table(key)
            diff += [im for im in want if im in table and table[im][0] != want[im][1]]
        if key == "mcships":
            diff += sorted(set(e["test_dedup"]) ^ set(read_split(folder, "mcships_testdedup") or []))
        if diff:
            bad += 1
            print("%-9s DIFFERS: %d images, e.g. %s (manifest %s, data %s)"
                  % (key, len(set(diff)), diff[0], want.get(diff[0]), have.get(diff[0])))
        else:
            print("%-9s ok: %d images%s" % (key, len(want), ", %d recordings" % len(e["recordings"]) if rec else ""))
    sys.exit(1 if bad else 0)


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "build"])
    ap.add_argument("--data", type=Path, default=DATA, help="the data folder (default: data/ of the repository)")
    a = ap.parse_args()
    DATA = a.data
    build() if a.cmd == "build" else check()


if __name__ == "__main__":
    main()
