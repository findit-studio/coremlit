//! What the model-gated suites measure in a child process: the process's
//! physical footprint, and running one gate alone in a child.

/// This process's physical footprint (`phys_footprint`), in bytes.
pub fn phys_footprint() -> u64 {
  // SAFETY: `task_vm_info` is plain-old-data; the all-zero bit pattern is a
  // valid value for every field.
  let mut info: mach2::task_info::task_vm_info = unsafe { core::mem::zeroed() };
  let mut count = (core::mem::size_of::<mach2::task_info::task_vm_info>()
    / core::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
  // SAFETY: `mach_task_self()` names the current task; `info` is a zeroed,
  // properly sized and aligned TASK_VM_INFO out-struct and `count` carries its
  // capacity in words per the in-out contract, so the kernel writes only
  // within `info`. The result code is checked before `info` is read.
  let result = unsafe {
    libc::task_info(
      mach2::traps::mach_task_self(),
      mach2::task_info::TASK_VM_INFO,
      (&raw mut info).cast(),
      &mut count,
    )
  };
  assert_eq!(result, libc::KERN_SUCCESS, "task_info(TASK_VM_INFO)");
  info.phys_footprint
}

/// Set in the child process a measurement runs in, naming what it measures.
pub const CHILD: &str = "EMBEDDINGGEMMA2_CHILD";

/// Runs `name` alone in a child process with `CHILD` set to `what`, and
/// returns the child's line that starts with `marker`.
pub fn in_child(name: &str, what: &str, marker: &str) -> String {
  let child = std::process::Command::new(std::env::current_exe().expect("the test binary"))
    .args([
      "--exact",
      name,
      "--ignored",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(CHILD, what)
    .output()
    .expect("the child runs");
  let stdout = String::from_utf8_lossy(&child.stdout).into_owned();
  assert!(
    child.status.success(),
    "the child failed:\n{stdout}\n{}",
    String::from_utf8_lossy(&child.stderr)
  );
  // libtest prints `test <name> ... ` on the line the child's own output
  // begins, so the marker is found within a line, not at its start.
  stdout
    .lines()
    .find_map(|line| line.split_once(marker).map(|(_, rest)| rest.to_owned()))
    .unwrap_or_else(|| panic!("the child printed no `{marker}` line:\n{stdout}"))
}
