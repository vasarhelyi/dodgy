//! Tests of avoiding neighbours with a known future path.

use super::*;

fn agent(position: Vec3, velocity: Vec3) -> Agent {
  Agent { position, velocity, radius: 1.0, avoidance_responsibility: 1.0 }
}

/// A neighbour that leaves all of the avoidance to the agent.
fn passive_neighbour(position: Vec3, velocity: Vec3) -> Agent {
  Agent { avoidance_responsibility: 0.0, ..agent(position, velocity) }
}

fn path<const N: usize>(points: [(f32, Vec3); N]) -> [PathPoint; N] {
  points.map(|(time, position)| PathPoint { time, position })
}

/// Finds the smallest distance between `agent` keeping `velocity` and
/// `neighbour` following `path`, until the end of the path.
fn min_distance(
  agent: &Agent,
  velocity: Vec3,
  neighbour: &Agent,
  path: &[PathPoint],
) -> f32 {
  let mut min_distance = f32::INFINITY;
  // Relative to the agent, each line of the path is from `start` to `end`.
  let mut start = neighbour.position - agent.position;
  for point in path {
    let end = point.position - agent.position - velocity * point.time;
    let line = end - start;
    let closest_ratio = if line == Vec3::ZERO {
      0.0
    } else {
      (-start.dot(line) / line.length_squared()).clamp(0.0, 1.0)
    };
    min_distance = min_distance.min((start + line * closest_ratio).length());
    start = end;
  }
  min_distance
}

mod get_plane_for_neighbour_path_tests {
  use rand::{rngs::StdRng, Rng, SeedableRng};

  use super::*;

  fn random_vec3(rng: &mut StdRng, size: f32) -> Vec3 {
    Vec3::from_array([(); 3].map(|_| rng.random_range(-size..size)))
  }

  #[test]
  fn straight_path_is_the_same_as_keeping_the_velocity() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..10000 {
      let agent = agent(random_vec3(&mut rng, 5.0), random_vec3(&mut rng, 5.0));
      let neighbour = Agent {
        avoidance_responsibility: rng.random_range(0.0..2.0),
        ..passive_neighbour(
          random_vec3(&mut rng, 5.0),
          random_vec3(&mut rng, 5.0),
        )
      };
      // The path can end both before and after the time horizon.
      let time_horizon = rng.random_range(0.5..10.0);
      let path = [1.0, 2.5, 5.0].map(|time| PathPoint {
        time,
        position: neighbour.position + neighbour.velocity * time,
      });

      let expected =
        agent.get_plane_for_neighbour(&neighbour, time_horizon, 0.1);
      let actual = agent.get_plane_for_neighbour_path(
        &neighbour,
        &path,
        time_horizon,
        0.1,
      );
      let distance_error = actual.signed_distance_to_plane(agent.velocity)
        - expected.signed_distance_to_plane(agent.velocity);
      assert!(
        distance_error.abs() < 1e-3
          && actual.normal.distance(expected.normal) < 1e-2,
        "{agent:?} {neighbour:?} {time_horizon}\n  left: {actual:?}\n right: \
         {expected:?}"
      );
    }
  }

  #[test]
  fn turning_path_only_restricts_colliding_velocities() {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..10000 {
      let agent =
        agent(random_vec3(&mut rng, 10.0), random_vec3(&mut rng, 3.0));
      let neighbour =
        passive_neighbour(random_vec3(&mut rng, 10.0), Vec3::ZERO);
      if agent.position.distance(neighbour.position) <= 2.0 {
        // Existing collisions ignore the path.
        continue;
      }
      let path = [1.0, 2.0, 3.0, 4.0]
        .map(|time| PathPoint { time, position: random_vec3(&mut rng, 10.0) });

      let plane =
        agent.get_plane_for_neighbour_path(&neighbour, &path, 4.0, 0.1);
      let is_allowed = plane.signed_distance_to_plane(agent.velocity) >= 0.0;
      let distance = min_distance(&agent, agent.velocity, &neighbour, &path);
      if distance < 1.999 {
        assert!(!is_allowed, "{distance} {agent:?} {neighbour:?} {path:?}");
      } else if distance > 2.001 {
        assert!(is_allowed, "{distance} {agent:?} {neighbour:?} {path:?}");
        // The plane is at the nearest colliding velocity.
        let distance = min_distance(&agent, plane.point, &neighbour, &path);
        assert!((distance - 2.0).abs() < 1e-2, "{distance} {agent:?} {path:?}");
      }
    }
  }

  #[test]
  fn neighbour_turning_away_does_not_restrict() {
    // Collides at the current velocities, but the neighbour turns sideways.
    let agent = agent(Vec3::ZERO, Vec3::X);
    let neighbour =
      passive_neighbour(Vec3::new(10.0, 0.5, 0.0), Vec3::new(-2.0, 0.0, 0.0));
    let path =
      path([(1.0, Vec3::new(8.0, 0.5, 0.0)), (5.0, Vec3::new(8.0, 12.5, 0.0))]);

    let straight = agent.get_plane_for_neighbour(&neighbour, 5.0, 0.1);
    assert!(straight.signed_distance_to_plane(agent.velocity) < 0.0);
    let turning =
      agent.get_plane_for_neighbour_path(&neighbour, &path, 5.0, 0.1);
    assert!(turning.signed_distance_to_plane(agent.velocity) > 0.0);
  }

  #[test]
  fn neighbour_turning_into_the_way_restricts() {
    // Parallel at the current velocities, but the neighbour turns towards the
    // agent.
    let agent = agent(Vec3::ZERO, Vec3::X);
    let neighbour = passive_neighbour(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
    let path =
      path([(1.0, Vec3::new(1.0, 5.0, 0.0)), (3.0, Vec3::new(3.0, 1.0, 0.0))]);

    let straight = agent.get_plane_for_neighbour(&neighbour, 3.0, 0.1);
    assert!(straight.signed_distance_to_plane(agent.velocity) > 0.0);
    let turning =
      agent.get_plane_for_neighbour_path(&neighbour, &path, 3.0, 0.1);
    assert!(turning.signed_distance_to_plane(agent.velocity) < 0.0);
  }

  #[test]
  fn last_line_continues_until_the_time_horizon() {
    // Collides after 8 seconds.
    let agent = agent(Vec3::ZERO, Vec3::ZERO);
    let neighbour = passive_neighbour(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO);
    let path = path([(1.0, Vec3::new(9.0, 0.0, 0.0))]);

    let before =
      agent.get_plane_for_neighbour_path(&neighbour, &path, 7.5, 0.1);
    assert!(before.signed_distance_to_plane(agent.velocity) > 0.0);
    let after = agent.get_plane_for_neighbour_path(&neighbour, &path, 8.5, 0.1);
    assert!(after.signed_distance_to_plane(agent.velocity) < 0.0);
  }

  #[test]
  fn path_after_the_time_horizon_is_ignored() {
    // Extending the second line back to the time horizon would go through the
    // agent.
    let agent = agent(Vec3::ZERO, Vec3::ZERO);
    let neighbour = passive_neighbour(Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO);
    let path = path([
      (2.0, Vec3::new(0.0, 10.0, 0.0)),
      (3.0, Vec3::new(0.0, 20.0, 0.0)),
    ]);

    let plane = agent.get_plane_for_neighbour_path(&neighbour, &path, 1.0, 0.1);
    assert!(plane.signed_distance_to_plane(agent.velocity) > 0.0);
  }

  #[test]
  fn avoids_head_on_neighbour() {
    let neighbour = passive_neighbour(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO);
    let path = path([(5.0, neighbour.position), (10.0, neighbour.position)]);

    // Reaches the neighbour at the time horizon, between the points, at a
    // point, and between a point and now.
    for speed in [1.0, 1.5, 2.0, 3.0] {
      let agent = agent(Vec3::ZERO, Vec3::X * speed);
      let plane =
        agent.get_plane_for_neighbour_path(&neighbour, &path, 10.0, 0.1);
      let distance = min_distance(&agent, plane.point, &neighbour, &path);
      assert!((distance - 2.0).abs() < 1e-3, "{speed} {distance} {plane:?}");
    }
  }

  #[test]
  fn existing_collision_ignores_the_path() {
    let agent = agent(Vec3::ZERO, Vec3::ZERO);
    let neighbour = passive_neighbour(Vec3::new(1.0, 1.0, 0.0), Vec3::ZERO);
    let path = path([(1.0, Vec3::new(20.0, 20.0, 0.0))]);

    let expected = agent.get_plane_for_neighbour(&neighbour, 1.0, 0.5);
    let actual =
      agent.get_plane_for_neighbour_path(&neighbour, &path, 1.0, 0.5);
    assert_eq!(actual.point, expected.point);
    assert_eq!(actual.normal, expected.normal);
  }

  #[test]
  #[should_panic(expected = "path times must be positive and increasing")]
  fn panics_if_path_times_are_not_increasing() {
    let agent = agent(Vec3::ZERO, Vec3::ZERO);
    let neighbour = passive_neighbour(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO);
    let path = path([(1.0, neighbour.position), (1.0, neighbour.position)]);
    agent.get_plane_for_neighbour_path(&neighbour, &path, 5.0, 0.1);
  }
}

mod project_to_line_obstacle_tests {
  use super::*;

  #[test]
  fn projects_to_the_largest_sphere_if_zero_is_inside_all() {
    // The largest sphere is at `start_time`, with a center of (1, 0, 0) and a
    // radius of 2.
    let (signed_distance, normal) = project_to_line_obstacle(
      /* origin= */ Vec3::new(2.0, 0.0, 0.0),
      /* relative_velocity= */ Vec3::new(-0.5, 0.0, 0.0),
      /* sum_radius= */ 4.0,
      /* start_time= */ 2.0,
      /* end_time= */ 4.0,
    );
    assert_eq!(signed_distance, -0.5);
    assert_eq!(normal, Vec3::NEG_X);
  }

  #[test]
  fn has_a_normal_in_the_center_of_the_sphere() {
    for origin in [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)] {
      let (signed_distance, normal) = project_to_line_obstacle(
        origin,
        /* relative_velocity= */ origin / 2.0,
        /* sum_radius= */ 4.0,
        /* start_time= */ 2.0,
        /* end_time= */ 4.0,
      );
      assert_eq!(signed_distance, -2.0);
      assert!(normal.is_normalized(), "{normal}");
    }
  }
}

mod compute_avoiding_velocity_with_paths_tests {
  use super::*;

  #[test]
  fn avoids_neighbour_on_its_path() {
    let agent = agent(Vec3::ZERO, Vec3::X);
    // Moves parallel to the agent, but its path turns towards the agent.
    let neighbour = passive_neighbour(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
    let path =
      path([(1.0, Vec3::new(1.0, 5.0, 0.0)), (3.0, Vec3::new(3.0, 1.0, 0.0))]);
    let compute = |path| {
      agent.compute_avoiding_velocity_with_paths(
        &[(Cow::Borrowed(&neighbour), path)],
        /* preferred_velocity= */ agent.velocity,
        /* max_speed= */ 2.0,
        /* time_step= */ 0.1,
        &AvoidanceOptions { time_horizon: 3.0 },
      )
    };

    // With an empty path, the neighbour keeps its velocity.
    assert_eq!(compute(&[]), agent.velocity);

    let distance = min_distance(&agent, compute(&path), &neighbour, &path);
    assert!(distance > 1.99, "{distance}");
  }

  #[cfg(feature = "debug")]
  #[test]
  fn debug_returns_a_plane_for_each_neighbour() {
    let agent = agent(Vec3::ZERO, Vec3::X);
    let neighbour = passive_neighbour(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
    let path = path([(3.0, Vec3::new(3.0, 1.0, 0.0))]);

    let (_, planes) = agent.compute_avoiding_velocity_with_paths_and_debug(
      &[(Cow::Borrowed(&neighbour), &path), (Cow::Borrowed(&neighbour), &[])],
      /* preferred_velocity= */ agent.velocity,
      /* max_speed= */ 2.0,
      /* time_step= */ 0.1,
      &AvoidanceOptions { time_horizon: 3.0 },
    );
    assert_eq!(planes.len(), 2);
    assert!(planes[0].signed_distance_to_plane(agent.velocity) < 0.0);
    assert!(planes[1].signed_distance_to_plane(agent.velocity) > 0.0);
  }
}
