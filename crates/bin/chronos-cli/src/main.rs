//! Chronos language compiler — command-line entry point.

use std::path::{Path, PathBuf};
use std::process::{Command, exit};

use chronos_compiler_core::{
    compile, CompilerConfig, CompilationTarget, TargetArch, TargetOS,
    SecurityMode, DiagnosticLevel,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { print_usage(); exit(0); }

    match args[1].as_str() {
        "build"    => cmd_build(&args[2..]),
        "check"    => cmd_check(&args[2..]),
        "run"      => cmd_run(&args[2..]),
        "version"  => cmd_version(),
        "--version" | "-V" => cmd_version(),
        "--help"   | "-h"  => print_usage(),
        unknown => {
            eprintln!("error: unknown command '{}'", unknown);
            exit(1);
        }
    }
}

fn print_usage() {
    println!("Chronos {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:  chronos <COMMAND> [OPTIONS] [FILE]");
    println!();
    println!("COMMANDS:");
    println!("  build [FILE]      Compile a .chr source file to a binary");
    println!("  check [FILE]      Type-check without producing output");
    println!("  run   [FILE]      Build and immediately execute");
    println!("  version           Print version info");
    println!();
    println!("OPTIONS:");
    println!("  -o <output>       Output file name  (default: same name as source)");
    println!("  --release         Optimised build");
    println!("  --emit-llvm       Write LLVM IR to <file>.ll and stop");
    println!("  --target <triple> Cross-compilation target  (x86_64|aarch64|wasm32)");
    println!("  --no-security     Disable security audit");
    println!();
    println!("EXAMPLE:");
    println!("  chronos build hello.chr");
    println!("  chronos run   hello.chr");
}

// ─────────────────────────────────────────────────────────────────────────────

struct BuildOptions {
    source_file: PathBuf,
    output_file: Option<PathBuf>,
    release: bool,
    emit_llvm: bool,
    target: CompilationTarget,
    security: bool,
    run_after: bool,
    run_args: Vec<String>,
}

fn parse_build_options(args: &[String]) -> BuildOptions {
    let mut source_file = None;
    let mut output_file = None;
    let mut release = false;
    let mut emit_llvm = false;
    let mut target = CompilationTarget {
        arch: TargetArch::X86_64,
        os: TargetOS::Linux,
        features: Vec::new(),
    };
    // Auto-detect host OS
    if cfg!(target_os = "windows") { target.os = TargetOS::Windows; }
    if cfg!(target_os = "macos")   { target.os = TargetOS::MacOS; }
    let mut security = true;
    let mut run_after = false;
    let mut run_args: Vec<String> = Vec::new();
    let mut i = 0;
    let mut past_double_dash = false;
    while i < args.len() {
        if past_double_dash { run_args.push(args[i].clone()); i += 1; continue; }
        match args[i].as_str() {
            "--release"     => release = true,
            "--emit-llvm"   => emit_llvm = true,
            "--no-security" => security = false,
            "--run"         => run_after = true,
            "--"            => past_double_dash = true,
            "-o" => { i += 1; if i < args.len() { output_file = Some(PathBuf::from(&args[i])); } }
            "--target" => {
                i += 1;
                if i < args.len() {
                    target = parse_target(&args[i]);
                }
            }
            f if !f.starts_with('-') => source_file = Some(PathBuf::from(f)),
            _ => {}
        }
        i += 1;
    }
    let source_file = source_file.unwrap_or_else(|| {
        // Look for a main.chr in the current directory
        PathBuf::from("main.chr")
    });
    BuildOptions { source_file, output_file, release, emit_llvm, target, security, run_after, run_args }
}

fn parse_target(s: &str) -> CompilationTarget {
    let (arch, os) = match s {
        "x86_64-linux"   | "x86_64-unknown-linux-gnu"  => (TargetArch::X86_64,  TargetOS::Linux),
        "x86_64-windows" | "x86_64-pc-windows-msvc"    => (TargetArch::X86_64,  TargetOS::Windows),
        "x86_64-macos"   | "x86_64-apple-darwin"       => (TargetArch::X86_64,  TargetOS::MacOS),
        "aarch64-linux"  | "aarch64-unknown-linux-gnu"  => (TargetArch::AArch64, TargetOS::Linux),
        "aarch64-macos"  | "aarch64-apple-darwin"       => (TargetArch::AArch64, TargetOS::MacOS),
        "wasm32"         | "wasm32-unknown-unknown"     => (TargetArch::Wasm32,  TargetOS::None),
        _ => { eprintln!("warning: unknown target '{}', defaulting to x86_64-linux", s); (TargetArch::X86_64, TargetOS::Linux) }
    };
    CompilationTarget { arch, os, features: Vec::new() }
}

// ─────────────────────────────────────────────────────────────────────────────

fn cmd_build(args: &[String]) {
    let opts = parse_build_options(args);
    let _ = do_build(&opts);
}

fn cmd_check(args: &[String]) {
    let opts = parse_build_options(args);
    let src = read_source(&opts.source_file);
    let config = make_config(&opts);
    let result = compile(src, opts.source_file.to_string_lossy().into_owned(), config);
    print_diagnostics(&result.diagnostics);
    if result.success {
        println!("No errors.");
    } else {
        exit(1);
    }
}

fn cmd_run(args: &[String]) {
    let mut opts = parse_build_options(args);
    opts.run_after = true;
    if let Some(binary) = do_build(&opts) {
        let status = Command::new(&binary)
            .args(&opts.run_args)
            .status()
            .unwrap_or_else(|e| { eprintln!("error: could not run '{}': {}", binary.display(), e); exit(1); });
        exit(status.code().unwrap_or(0));
    }
}

fn cmd_version() {
    println!("chronos {}", env!("CARGO_PKG_VERSION"));
    println!("host: {}", current_triple());
    // Check if MSVC is available on Windows
    if cfg!(windows) {
        if let Some((cl_exe, _)) = find_msvc() {
            println!("msvc: {}", cl_exe.display());
        } else {
            println!("msvc: not found");
        }
    }
    // Check if clang is available
    if let Ok(out) = Command::new("clang").arg("--version").output() {
        let ver = String::from_utf8_lossy(&out.stdout);
        println!("clang: {}", ver.lines().next().unwrap_or("available"));
    } else {
        println!("clang: not found");
    }
}

fn current_triple() -> &'static str {
    if cfg!(target_os = "windows") { "x86_64-windows" }
    else if cfg!(target_os = "macos") { "aarch64-macos" }
    else { "x86_64-linux" }
}

// ─────────────────────────────────────────────────────────────────────────────
// MSVC discovery and invocation
// ─────────────────────────────────────────────────────────────────────────────

fn find_msvc() -> Option<(PathBuf, PathBuf)> {
    // Returns (cl_exe_path, msvc_base_dir) if MSVC is found.
    // msvc_base_dir is the directory containing include/ and lib/.
    let candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64\cl.exe",
    ];
    for path in &candidates {
        let p = Path::new(path);
        if p.exists() {
            // cl.exe lives at: <base>\bin\Hostx64\x64\cl.exe
            // so <base> is four parents up
            if let Some(base) = p.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                return Some((p.to_path_buf(), base.to_path_buf()));
            }
        }
    }
    None
}

fn find_winsdk_version() -> Option<String> {
    let lib_dir = Path::new(r"C:\Program Files (x86)\Windows Kits\10\Lib");
    if let Ok(entries) = std::fs::read_dir(lib_dir) {
        let mut versions: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        versions.sort_by(|a, b| b.cmp(a)); // newest first
        return versions.into_iter().next();
    }
    None
}

fn invoke_msvc(c_file: &Path, out: &Path, release: bool) -> bool {
    let Some((cl_exe, msvc_base)) = find_msvc() else {
        eprintln!("warning: MSVC (cl.exe) not found");
        return false;
    };
    let sdk_ver = find_winsdk_version().unwrap_or_else(|| "10.0.26100.0".to_string());

    let msvc_include = msvc_base.join("include");
    let msvc_lib     = msvc_base.join("lib").join("x64");
    let sdk_base     = Path::new(r"C:\Program Files (x86)\Windows Kits\10");
    let ucrt_inc     = sdk_base.join("Include").join(&sdk_ver).join("ucrt");
    let um_inc       = sdk_base.join("Include").join(&sdk_ver).join("um");
    let shared_inc   = sdk_base.join("Include").join(&sdk_ver).join("shared");
    let ucrt_lib     = sdk_base.join("Lib").join(&sdk_ver).join("ucrt").join("x64");
    let um_lib       = sdk_base.join("Lib").join(&sdk_ver).join("um").join("x64");

    let mut cmd = Command::new(&cl_exe);
    cmd.arg(c_file);
    cmd.arg(format!("/Fe:{}", out.display()));
    cmd.arg("/nologo");
    cmd.arg("/std:c11");
    if release { cmd.arg("/O2"); } else { cmd.arg("/Od"); }
    // Include paths
    cmd.arg(format!("/I{}", msvc_include.display()));
    cmd.arg(format!("/I{}", ucrt_inc.display()));
    cmd.arg(format!("/I{}", um_inc.display()));
    cmd.arg(format!("/I{}", shared_inc.display()));
    // Linker flags
    cmd.arg("/link");
    cmd.arg(format!("/LIBPATH:{}", msvc_lib.display()));
    cmd.arg(format!("/LIBPATH:{}", ucrt_lib.display()));
    cmd.arg(format!("/LIBPATH:{}", um_lib.display()));

    // Set INCLUDE/LIB env vars (cl.exe also reads these)
    let include_env = format!("{};{};{};{}",
        msvc_include.display(), ucrt_inc.display(),
        um_inc.display(), shared_inc.display());
    let lib_env = format!("{};{};{}",
        msvc_lib.display(), ucrt_lib.display(), um_lib.display());
    cmd.env("INCLUDE", &include_env);
    cmd.env("LIB", &lib_env);

    match cmd.status() {
        Ok(s) if s.success() => true,
        Ok(s) => { eprintln!("error: cl.exe exited with status {}", s); false }
        Err(e) => { eprintln!("error: failed to run cl.exe: {}", e); false }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-Windows C compiler fallback
// ─────────────────────────────────────────────────────────────────────────────

fn invoke_c_compiler(c_file: &Path, out: &Path, release: bool) -> bool {
    for compiler in &["cc", "gcc", "clang"] {
        let mut cmd = Command::new(compiler);
        cmd.arg("-std=c11").arg(c_file).arg("-o").arg(out).arg("-lm");
        if release { cmd.arg("-O2"); }
        if let Ok(s) = cmd.status() {
            if s.success() { return true; }
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────

fn do_build(opts: &BuildOptions) -> Option<PathBuf> {
    let src = read_source(&opts.source_file);
    let stem = opts.source_file.file_stem()
        .unwrap_or_default().to_string_lossy().into_owned();

    let config = make_config(opts);
    eprint!("  Compiling {} ...", opts.source_file.display());
    let result = compile(src, opts.source_file.to_string_lossy().into_owned(), config);
    eprintln!(" done");

    print_diagnostics(&result.diagnostics);

    if !result.success {
        eprintln!("error: compilation failed");
        exit(1);
    }

    let binary = opts.output_file.clone().unwrap_or_else(|| {
        if cfg!(windows) { PathBuf::from(format!("{}.exe", stem)) }
        else { PathBuf::from(&stem) }
    });

    // ── Try C backend first (most portable) ──────────────────────────────────
    if let Some(c_code) = &result.c_code {
        if !c_code.is_empty() {
            let c_path = PathBuf::from(format!("{}.c", stem));
            std::fs::write(&c_path, c_code).unwrap_or_else(|e| {
                eprintln!("error writing {}: {}", c_path.display(), e);
                exit(1);
            });

            let compiled = if cfg!(windows) {
                invoke_msvc(&c_path, &binary, opts.release)
            } else {
                invoke_c_compiler(&c_path, &binary, opts.release)
            };

            if compiled {
                println!("  Binary: {}", binary.display());
                return Some(binary);
            } else {
                eprintln!("C source saved to: {}", c_path.display());
                if cfg!(windows) {
                    eprintln!("Compile manually: cl.exe /Fe:{} {}", binary.display(), c_path.display());
                } else {
                    eprintln!("Compile manually: cc -std=c11 {} -o {} -lm", c_path.display(), binary.display());
                }
                return None;
            }
        }
    }

    // ── Fall back to LLVM IR ──────────────────────────────────────────────────
    let llvm_ir = match &result.llvm_ir {
        Some(ir) if !ir.is_empty() => ir.clone(),
        _ => {
            eprintln!("error: compiler produced no output (pipeline incomplete for this input)");
            eprintln!("hint: the lowering pass currently supports fn/struct/enum declarations.");
            exit(1);
        }
    };

    // Write .ll file
    let ll_path = PathBuf::from(format!("{}.ll", stem));
    std::fs::write(&ll_path, &llvm_ir)
        .unwrap_or_else(|e| { eprintln!("error: could not write {}: {}", ll_path.display(), e); exit(1); });

    if opts.emit_llvm {
        println!("LLVM IR written to {}", ll_path.display());
        return None;
    }

    if !invoke_clang(&ll_path, &binary, opts.release) {
        eprintln!();
        eprintln!("LLVM IR has been saved to: {}", ll_path.display());
        eprintln!("To compile manually once clang is installed:");
        eprintln!("  clang -x ir {} -o {}", ll_path.display(), binary.display());
        return None;
    }

    println!("  Binary: {}", binary.display());
    Some(binary)
}

fn invoke_clang(ll: &Path, out: &Path, release: bool) -> bool {
    let mut cmd = Command::new("clang");
    cmd.args(["-x", "ir"]);
    cmd.arg(ll);
    cmd.args(["-o", out.to_str().unwrap()]);
    if release { cmd.arg("-O2"); }
    // Link with libc so print/printf work
    if cfg!(target_os = "linux")   { cmd.args(["-lm", "-lpthread"]); }
    if cfg!(target_os = "windows") { cmd.args(["-lmsvcrt"]); }
    match cmd.status() {
        Ok(s) if s.success() => true,
        Ok(s) => {
            eprintln!("error: clang exited with status {}", s);
            false
        }
        Err(_) => {
            eprintln!("warning: clang not found — skipping native compilation");
            false
        }
    }
}

fn read_source(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: could not read '{}': {}", path.display(), e);
        exit(1);
    })
}

fn make_config(opts: &BuildOptions) -> CompilerConfig {
    CompilerConfig {
        security_mode: if opts.security { SecurityMode::Warn } else { SecurityMode::Off },
        realtime_mode: false,
        target: opts.target.clone(),
        hardware_model: None,
    }
}

fn print_diagnostics(diags: &[chronos_compiler_core::CompilerDiagnostic]) {
    for d in diags {
        let prefix = match d.level {
            DiagnosticLevel::Error   => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Info    => "info",
            DiagnosticLevel::Hint    => "hint",
        };
        eprintln!("{} [{}]: {}", prefix, d.code, d.message);
    }
}
