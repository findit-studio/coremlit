use super::*;

fn refused_at(result: Result<()>) -> Option<(AllocationSite, usize)> {
  match result {
    Err(Error::Allocation(refusal)) => Some((refusal.site(), refusal.bytes())),
    _ => None,
  }
}

/// Every door reserves when the seam is quiet, and refuses by name, with the
/// bytes asked for, when it is set to the door's site.
#[test]
fn every_door_refuses_by_name_under_the_seam() {
  let site = AllocationSite::GatherRows;
  let mut list: Vec<u32> = Vec::new();
  reserve(&mut list, 4, site).expect("quiet");
  assert!(list.capacity() >= 4);
  let _guard = inject::refuse(site, 0);
  assert_eq!(refused_at(reserve(&mut list, 8, site)), Some((site, 32)));
  assert_eq!(
    refused_at(reserve_exact(&mut list, 8, site)),
    Some((site, 32))
  );
  let mut heap: BinaryHeap<u64> = BinaryHeap::new();
  assert_eq!(
    refused_at(reserve_heap(&mut heap, 3, site)),
    Some((site, 24))
  );
  let mut text = String::new();
  assert_eq!(refused_at(reserve_str(&mut text, 5, site)), Some((site, 5)));
  assert!(matches!(
    with_capacity::<u16>(2, site),
    Err(Error::Allocation(a)) if a.site() == site && a.bytes() == 4
  ));
  assert!(matches!(
    collect([1u8, 2, 3].into_iter(), site),
    Err(Error::Allocation(a)) if a.site() == site
  ));
}

/// The seam refuses only its own site, only after its skips, and clears when
/// its guard drops.
#[test]
fn the_seam_counts_its_skips_and_clears() {
  let (site, other) = (AllocationSite::BatchOutput, AllocationSite::PackLayout);
  let mut list: Vec<u8> = Vec::new();
  {
    let _guard = inject::refuse(site, 2);
    reserve(&mut list, 1, other).expect("another site is not refused");
    reserve(&mut list, 1, site).expect("the first skip");
    reserve(&mut list, 1, site).expect("the second skip");
    assert!(refused_at(reserve(&mut list, 1, site)).is_some());
    assert!(refused_at(reserve(&mut list, 1, site)).is_some());
  }
  reserve(&mut list, 1, site).expect("the guard cleared the seam");
}

/// A request no allocator can hold is refused by name without the seam.
#[test]
fn an_impossible_request_is_refused_by_name() {
  let mut list: Vec<u64> = Vec::new();
  assert_eq!(
    refused_at(reserve(&mut list, usize::MAX, AllocationSite::WindowList)),
    Some((AllocationSite::WindowList, usize::MAX))
  );
}
