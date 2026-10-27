"""Local model lifecycle and verified public GGUF acquisition (standard library)."""
import ctypes
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import rag_local as rag

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ALLOWED_AUTHORS = ("Qwen", "ggml-org", "bartowski", "mradermacher", "unsloth", "lmstudio-community")


class ModelStartupError(RuntimeError):
    pass


def read_url(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "PatrimoTripAI-local-benchmark/1.0"}), timeout=60) as response:
        return json.load(response)


def compatible_card(card):
    if not isinstance(card, dict):
        return False
    base = card.get("base_model", [])
    if isinstance(base, str):
        base = [base]
    return card.get("license") == "apache-2.0" and isinstance(base, list) and BASE_MODEL in base


def choose_file(files):
    for quant in ("Q8_0", "Q4_K_M"):
        candidates = [f for f in files if re.search(rf"(?:^|[-_.]){quant}\.gguf$", f.get("rfilename", ""), re.I)
                      and (f.get("lfs") or {}).get("sha256") and 0 < (f.get("lfs") or {}).get("size", 0) < 1_500_000_000
                      and ".." not in pathlib.PurePosixPath(f["rfilename"]).parts]
        if candidates:
            return sorted(candidates, key=lambda f: f["rfilename"])[0]
    raise ValueError("No verified single-file Q8_0 or Q4_K_M embedding GGUF found.")


def download_embedding(folder):
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "model.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = folder / pathlib.PurePosixPath(metadata["filename"]).name
        if compatible_card(metadata) and path.is_file() and rag.sha_file(path) == metadata["sha256"]:
            return path, metadata
    print("Recherche du modèle d'embeddings public et de sa licence...", flush=True)
    params = urllib.parse.urlencode({"search": "Qwen3-Embedding-0.6B-GGUF", "limit": 100})
    repos = read_url("https://huggingface.co/api/models?" + params)
    repos = [r for r in repos if r.get("id", "").split("/")[0] in ALLOWED_AUTHORS]
    repos.sort(key=lambda r: (ALLOWED_AUTHORS.index(r["id"].split("/")[0]), r["id"]))
    errors = []
    for repository in repos[:12]:
        repo = repository["id"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            continue
        try:
            info = read_url(f"https://huggingface.co/api/models/{repo}?blobs=true")
            card = info.get("cardData", {})
            if not compatible_card(card):
                continue
            selected = choose_file(info.get("siblings", []))
            revision = info["sha"]
            expected = selected["lfs"]["sha256"]
            if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Invalid immutable revision or checksum.")
            name = pathlib.PurePosixPath(selected["rfilename"]).name
            destination = folder / name
            if destination.exists() and rag.sha_file(destination) != expected:
                raise ValueError("A different model file exists; it was not overwritten.")
            if not destination.exists():
                url = f"https://huggingface.co/{repo}/resolve/{revision}/" + urllib.parse.quote(selected["rfilename"], safe="/")
                temporary = destination.with_suffix(".gguf.part")
                print(f"Téléchargement vérifié : {repo}/{name}", flush=True)
                digest = hashlib.sha256()
                transferred, next_progress = 0, 32*1024*1024
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "PatrimoTripAI-local-benchmark/1.0"}), timeout=90) as response, temporary.open("wb") as output:
                    while block := response.read(1024*1024):
                        output.write(block)
                        digest.update(block)
                        transferred += len(block)
                        if transferred >= next_progress:
                            print(f"Modèle : {transferred//1024**2}/{selected['lfs']['size']//1024**2} MiB", flush=True)
                            next_progress += 32*1024*1024
                if digest.hexdigest() != expected:
                    raise ValueError("Model checksum mismatch; partial file not used.")
                temporary.replace(destination)
            metadata = {"repository": repo, "revision": revision, "filename": selected["rfilename"],
                        "sha256": expected, "license": card["license"], "base_model": card["base_model"],
                        "size": selected["lfs"]["size"], "pooling": "last"}
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            return destination, metadata
        except (ValueError, OSError, KeyError) as error:
            errors.append(f"{repo}: {error}")
    raise RuntimeError("No downloadable model passed the base-model/license/checksum checks. " + "; ".join(errors[-3:]))


def windows_environment():
    environment = os.environ.copy()
    if os.name == "nt":
        import winreg
        paths = [environment.get("PATH", "")]
        for hive, key in [(winreg.HKEY_CURRENT_USER, "Environment"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]:
            try:
                with winreg.OpenKey(hive, key) as opened:
                    paths.append(os.path.expandvars(winreg.QueryValueEx(opened, "Path")[0]))
            except OSError:
                pass
        environment["PATH"] = os.pathsep.join(paths)
    return environment


def port_busy(port):
    with socket.socket() as connection:
        connection.settimeout(0.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def is_expected_test_process(process, generation_model):
    command = process.get("CommandLine") or ""
    return (process.get("Name", "").casefold() == "llama-server.exe"
            and str(generation_model).casefold() in command.casefold()
            and re.search(r"--port(?:\s+|=)8080(?:\s|$)", command)
            and re.search(r"--host(?:\s+|=)127\.0\.0\.1(?:\s|$)", command))


class Models:
    def __init__(self, generation_model, projector, logs, takeover=False):
        self.environment = windows_environment()
        self.executable = shutil.which("llama-server", path=self.environment.get("PATH"))
        if not self.executable:
            raise RuntimeError("llama-server is not available in PATH.")
        self.model, self.projector, self.logs = generation_model, projector, logs
        self.takeover, self.process, self.port = takeover, None, None
        self.events = []

    def event(self, phase, **data):
        self.events.append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": phase, **data})

    def prepare_ports(self):
        if port_busy(8081):
            raise RuntimeError("Port 8081 is already occupied; no process was stopped.")
        if not port_busy(8080):
            return
        if not self.takeover or os.name != "nt":
            raise RuntimeError("Port 8080 is busy. Close the existing test server, or use --take-over-test-server on Windows.")
        script = """[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false);
$ids = @(Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique);
$items = @($ids | ForEach-Object { Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_) | Select-Object ProcessId,Name,CommandLine });
ConvertTo-Json -InputObject $items -Compress
"""
        found = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True,
                               encoding="utf-8-sig", errors="replace", timeout=20, check=True, env=self.environment)
        processes = json.loads(found.stdout)
        if isinstance(processes, dict):
            processes = [processes]
        if not processes or not all(is_expected_test_process(p, self.model) for p in processes):
            raise RuntimeError("Port 8080 does not belong to the expected local 4B test server. Nothing was stopped.")
        for process in processes:
            pid = int(process["ProcessId"])
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -ErrorAction Stop"],
                           check=True, timeout=15, env=self.environment, capture_output=True)
        for _ in range(30):
            if not port_busy(8080):
                self.event("adopted_known_test_server")
                return
            time.sleep(0.2)
        raise RuntimeError("The former test server has not released its port.")

    def start(self, role, embedding_model=None, embedding_cpu=False, generation_vision=True):
        self.stop()
        port = 8081 if role == "embedding" else 8080
        if port_busy(port):
            raise RuntimeError(f"Port {port} is occupied by another process; refusing to replace it.")
        common = ["--host", "127.0.0.1", "--port", str(port), "--device", "none" if embedding_cpu else "Vulkan0", "--parallel", "1"]
        if role == "embedding":
            args = ["--model", str(embedding_model), "--embedding", "--pooling", "last", "--n-gpu-layers", "0" if embedding_cpu else "99",
                    "--ctx-size", "2048", "--batch-size", "2048", "--ubatch-size", "2048"]
        else:
            args = ["--model", str(self.model),
                    "--n-gpu-layers", "20", "--ctx-size", "4096", "--batch-size", "256", "--ubatch-size", "128", "--reasoning-budget", "0"]
            if generation_vision:
                args += ["--mmproj", str(self.projector), "--mmproj-offload"]
        print(f"Chargement local : {role} (un seul modèle à la fois)...", flush=True)
        started = time.perf_counter()
        log = self.logs / f"{role}-{uuid_suffix()}.log"
        with log.open("wb") as stream:
            self.process = subprocess.Popen([self.executable, *args, *common], stdout=stream, stderr=subprocess.STDOUT, env=self.environment)
        self.port = port
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(240):
            if self.process.poll() is not None:
                raise ModelStartupError(f"{role} startup failed. See {log.name}.")
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        self.event(role + "_ready", startup_seconds=time.perf_counter()-started, embedding_cpu=embedding_cpu,
                                   generation_vision=generation_vision if role == "generation" else None)
                        return f"http://127.0.0.1:{port}"
            except (OSError, ValueError):
                pass
            if attempt and attempt % 30 == 0:
                print(f"Initialisation {role} en cours...", flush=True)
            time.sleep(0.5)
        raise ModelStartupError(f"Timeout loading {role}; see {log.name}.")

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.event("managed_process_stopped", port=self.port)
            self.process, self.port = None, None


def uuid_suffix():
    import uuid
    return uuid.uuid4().hex[:8]


def available_memory_mib():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in ["total", "available", "page_total", "page_free", "virtual_total", "virtual_free", "extended"]]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return None
        return round(memory.available/1024**2, 1)
    try:
        values = dict(line.split(":", 1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines())
        return round(int(values["MemAvailable"].split()[0])/1024, 1)
    except (OSError, ValueError, KeyError):
        return None


class Resources:
    def __init__(self, folder, environment):
        self.folder, self.environment = folder, environment
        self.phase, self.stop_event, self.thread, self.gpu = "setup", threading.Event(), None, None

    def start(self):
        def sample():
            with (self.folder / "memory.jsonl").open("w", encoding="utf-8") as stream:
                while not self.stop_event.is_set():
                    stream.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": self.phase,
                                             "available_ram_mib": available_memory_mib()})+"\n")
                    stream.flush()
                    self.stop_event.wait(1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        smi = shutil.which("nvidia-smi", path=self.environment.get("PATH"))
        if smi:
            with (self.folder / "gpu.csv").open("wb") as output, (self.folder / "gpu-errors.txt").open("wb") as error:
                self.gpu = subprocess.Popen([smi, "--query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu", "--format=csv", "-l", "1"],
                                            stdout=output, stderr=error, env=self.environment)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.gpu and self.gpu.poll() is None:
            self.gpu.terminate()
            try:
                self.gpu.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gpu.kill()
                self.gpu.wait(timeout=5)
