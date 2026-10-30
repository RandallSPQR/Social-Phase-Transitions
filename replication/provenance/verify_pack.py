"""Verify the original-history pack against the pre-registration provenance table, in a fresh repository.

  python replication/provenance/verify_pack.py PACK_DIR

PACK_DIR holds original-history-<id>.pack and pack_manifest.json (from `build_provenance.py pack`, or from the Zenodo
archive). The script:
  1. creates an empty bare repository and runs `git index-pack` on the pack. Git recomputes the SHA-1 of every object
     it unpacks, so every commit, tree and blob hash in the pack is verified from content;
  2. checks the pack holds no PDF and no OpenRouter generation id (the omitted material);
  3. for every row of provenance.json: the commit exists, the pre-registration document at that commit hashes to the
     recorded SHA-256, and the commit is an ancestor of the commit GitHub's push log records for that push.
The repository's root commit (an empty README made by GitHub, carrying a personal email) is also left out; its
children are history boundaries. Blobs under papers/ and results/llm/cache/ are deliberately absent: commits and trees reference them by hash only, so
step 3 does not need them. `git fsck` therefore reports them missing; that is expected.
"""
import glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(pack_dir):
    prov = json.load(open(os.path.join(HERE, "provenance.json")))["rows"]
    gh = {e["id"]: e for e in json.load(open(os.path.join(HERE, "github_push_log.json")))["events"]}
    pack = glob.glob(os.path.join(pack_dir, "original-history-*.pack"))[0]
    repo = tempfile.mkdtemp(prefix="verify_")
    g = lambda *a, **k: subprocess.run(["git", "-C", repo, *a], capture_output=True, check=k.get("check", True), **({"input": k["input"]} if "input" in k else {}))
    subprocess.run(["git", "init", "-q", "--bare", repo], check=True)
    dst = os.path.join(repo, "objects", "pack", os.path.basename(pack)); shutil.copy(pack, dst)
    g("index-pack", dst)                                   # recomputes and verifies every object hash
    print("index-pack: every object hash verified from content")
    man = json.load(open(os.path.join(pack_dir, "pack_manifest.json")))
    # commits left out of the pack (pack_manifest.json, commits_omitted): their children are history boundaries
    open(os.path.join(repo, "shallow"), "w").write("".join(c + "\n" for c in man.get("shallow", [])))

    # 2. omitted material really is absent
    objs = g("cat-file", "--batch-all-objects", "--batch-check").stdout.decode().split("\n")
    blobs = [l.split()[0] for l in objs if l.endswith(tuple("0123456789")) and " blob " in l]
    bad_pdf = bad_gen = 0
    for b in blobs:
        data = g("cat-file", "blob", b).stdout
        bad_pdf += data[:5] == b"%PDF-"
        bad_gen += bool(re.search(rb"gen-[0-9]{10}-[A-Za-z0-9]{10,}", data))
    print(f"blobs in pack: {len(blobs)}; PDFs: {bad_pdf}; blobs with generation ids: {bad_gen}")
    commits = [l.split()[0] for l in objs if " commit " in l]
    for c in commits:                                      # no personal email in any commit header
        bad_gen += bool(re.search(rb"<(?!noreply@)[^>\s]*@[^>\s]*>", g("cat-file", "commit", c).stdout))
    print(f"commits in pack: {len(commits)}; only noreply addresses: {'yes' if not bad_gen else 'NO'}")

    # 3. every pre-registration document as committed
    fails = 0
    for r in prov:
        path = r["document"].split(" ")[0]
        txt = g("show", f"{r['commit']}:{path}").stdout
        ok_hash = hashlib.sha256(txt).hexdigest() == r["file_sha256"]
        ev = gh.get(r["github_push_event"]) or {}
        anc = g("merge-base", "--is-ancestor", r["commit"], ev.get("after", ""), check=False).returncode == 0
        fails += not (ok_hash and anc)
        print(f"{'ok ' if ok_hash and anc else 'BAD'} {r['short']} {path:<36} sha256 {'ok' if ok_hash else 'MISMATCH'}; "
              f"in GitHub push {ev.get('timestamp')} {'ok' if anc else 'NOT FOUND'}")
    shutil.rmtree(repo)
    print("ALL VERIFIED" if not (fails or bad_pdf or bad_gen) else "VERIFICATION FAILED")
    sys.exit(1 if fails or bad_pdf or bad_gen else 0)


if __name__ == "__main__":
    main(sys.argv[1])
