//! Location & time tab for the debug UI.
//!
//! Provides geocoding search, coordinate input, altitude control, and time-of-day settings.

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::system::SystemParam,
    prelude::*,
};
use bevy_egui::egui;
use glam::DVec3;

use crate::{
    async_runtime::TaskSpawner,
    camera::{AltitudeRequest, FlightCamera, HeadingRequest, TranslateRequest},
    world::{
        coords::ecef_to_lat_lon,
        geo::{
            GEOCODING_THROTTLE_SECS, GeocodingState, HttpClient, TeleportAnimation, TeleportState,
        },
        moon::compute_moon_state,
        time_of_day::{SECONDS_PER_HOUR, TimeMode, TimeOfDayState, local_to_utc, seconds_to_hms},
    },
};

/// State for the lat/long text input fields.
#[derive(Resource)]
pub(super) struct CoordinateInputState {
    lat_text: String,
    lon_text: String,
    /// Track whether text fields are focused to avoid overwriting user input.
    is_editing: bool,
    /// Selected distance (metres) for the precise-translation buttons.
    translate_distance_m: f64,
}

impl Default for CoordinateInputState {
    fn default() -> Self {
        Self {
            lat_text: String::new(),
            lon_text: String::new(),
            is_editing: false,
            translate_distance_m: 1000.0,
        }
    }
}

/// Resources for the location & time tab.
#[derive(SystemParam)]
pub(super) struct LocationParams<'w, 's> {
    pub coord_state: ResMut<'w, CoordinateInputState>,
    pub geocoding_state: ResMut<'w, GeocodingState>,
    pub teleport_state: ResMut<'w, TeleportState>,
    pub teleport_animation: Res<'w, TeleportAnimation>,
    pub time_of_day: ResMut<'w, TimeOfDayState>,
    pub http_client: Res<'w, HttpClient>,
    pub spawner: TaskSpawner<'w, 's>,
    pub altitude_request: ResMut<'w, AltitudeRequest>,
    pub heading_request: ResMut<'w, HeadingRequest>,
    pub translate_request: ResMut<'w, TranslateRequest>,
    /// Read-only — coexists with the camera tab's read-only flight-camera
    /// query in the same system. Heading changes flow back through
    /// [`HeadingRequest`].
    pub flight_camera_query: Query<'w, 's, &'static FlightCamera>,
    pub diagnostics: Res<'w, DiagnosticsStore>,
}

/// Render the location & time tab content and execute any resulting actions.
pub(super) fn render_location_tab(
    ui: &mut egui::Ui,
    time: &Time,
    location: &mut LocationParams,
    position: DVec3,
) {
    let fps = location
        .diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed)
        .unwrap_or(0.0);
    ui.label(format!(
        "FPS: {fps:.0}  ·  Position: ({:.0}, {:.0}, {:.0})",
        position.x, position.y, position.z
    ));
    ui.separator();

    let (lat_deg, lon_deg) = ecef_to_lat_lon(position);
    let altitude = position.length() - crate::constants::EARTH_RADIUS_M_F64;

    // Update text fields when not editing and not teleporting.
    if !location.coord_state.is_editing && !location.teleport_state.is_pending() {
        location.coord_state.lat_text = format!("{lat_deg:.6}");
        location.coord_state.lon_text = format!("{lon_deg:.6}");
    }

    let mut start_geocoding = false;
    let mut start_reverse_geocoding = false;
    let mut new_coords: Option<(f64, f64)> = None;

    // Geocoding search.
    ui.horizontal(|ui| {
        ui.label("Search:");
        let response = ui.add(
            egui::TextEdit::singleline(&mut location.geocoding_state.search_text)
                .desired_width(150.0)
                .hint_text("City, address..."),
        );
        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            start_geocoding = true;
        }
        if ui.button("Go").clicked() {
            start_geocoding = true;
        }
        if ui
            .button("Here?")
            .on_hover_text("Look up current location")
            .clicked()
        {
            start_reverse_geocoding = true;
        }
    });

    // Show loading/throttle status.
    let current_time = time.elapsed_secs_f64();
    if location.geocoding_state.is_loading {
        ui.label("Searching...");
    } else if let Some(last_time) = location.geocoding_state.last_request_time {
        let elapsed = current_time - last_time;
        if elapsed < GEOCODING_THROTTLE_SECS {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let remaining = (GEOCODING_THROTTLE_SECS - elapsed).ceil() as u64;
            ui.label(format!("Wait {remaining}s before next search"));
        }
    }

    // Show geocoding error if any.
    if let Some(ref error) = location.geocoding_state.error {
        ui.colored_label(egui::Color32::RED, error);
    }

    // Show results.
    if !location.geocoding_state.results.is_empty() {
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
                for result in &location.geocoding_state.results {
                    if ui.link(&result.display_name).clicked() {
                        new_coords = Some((result.lat, result.lon));
                    }
                }
            });
    }

    // Nominatim attribution (required by usage policy).
    ui.separator();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.label("Search by");
        ui.hyperlink_to("Nominatim", "https://nominatim.openstreetmap.org/");
        ui.label("\u{00a9} OpenStreetMap");
    });

    ui.separator();

    // Show teleport status.
    if location.teleport_animation.is_waiting_for_physics() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Waiting for terrain to load...");
        });
        ui.add(egui::ProgressBar::new(1.0).show_percentage());
    } else if let Some(progress) = location.teleport_animation.progress() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Flying...");
        });
        ui.add(egui::ProgressBar::new(progress).show_percentage());
    } else if location.teleport_state.is_pending() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Fetching elevation...");
        });
    } else if let Some(ref error) = location.teleport_state.error {
        ui.colored_label(egui::Color32::RED, format!("Teleport failed: {error}"));
    }

    // Lat/lon input fields on the same row.
    ui.horizontal(|ui| {
        ui.label("Lat:");
        let lat_response = ui.add(
            egui::TextEdit::singleline(&mut location.coord_state.lat_text).desired_width(80.0),
        );
        if lat_response.has_focus() {
            location.coord_state.is_editing = true;
        }
        if lat_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let (Ok(lat), Ok(lon)) = (
                location.coord_state.lat_text.parse::<f64>(),
                location.coord_state.lon_text.parse::<f64>(),
            ) {
                new_coords = Some((lat.clamp(-90.0, 90.0), lon));
            }
            location.coord_state.is_editing = false;
        }

        ui.label("Lon:");
        let lon_response = ui.add(
            egui::TextEdit::singleline(&mut location.coord_state.lon_text).desired_width(80.0),
        );
        if lon_response.has_focus() {
            location.coord_state.is_editing = true;
        }
        if lon_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let (Ok(lat), Ok(lon)) = (
                location.coord_state.lat_text.parse::<f64>(),
                location.coord_state.lon_text.parse::<f64>(),
            ) {
                new_coords = Some((lat.clamp(-90.0, 90.0), lon));
            }
            location.coord_state.is_editing = false;
        }
    });

    // Altitude slider (logarithmic scale from 1m to 10,000km).
    let mut slider_alt = altitude.clamp(1.0, 10_000_000.0);
    ui.horizontal(|ui| {
        ui.label("Alt:");
        if ui
            .add(
                egui::Slider::new(&mut slider_alt, 1.0..=10_000_000.0)
                    .logarithmic(true)
                    .update_while_editing(false)
                    .suffix(" m"),
            )
            .changed()
        {
            location.altitude_request.request(slider_alt);
        }
    });

    // Compass: shows the camera's yaw relative to local north (the
    // tangent direction toward the world +Z pole). Useful for aligning
    // with cardinal axes when reasoning about parallax / wind / shadow
    // direction; heading edits route through `HeadingRequest` so the
    // applier system can update the camera entity in a single
    // disjoint-borrow place.
    render_compass(
        ui,
        &mut location.heading_request,
        &location.flight_camera_query,
        position,
    );

    // Precise translation: move the camera an exact great-circle
    // distance along a cardinal direction. Repeatable (unlike
    // hand-flown movement), so diagnostics can correlate a known
    // displacement with the resulting drift.
    render_translation_controls(
        ui,
        &mut location.coord_state.translate_distance_m,
        &mut location.translate_request,
    );

    ui.separator();

    // Time of day controls.
    ui.horizontal(|ui| {
        ui.label("Time of day:");
        if ui
            .selectable_label(location.time_of_day.mode == TimeMode::Realtime, "Realtime")
            .clicked()
        {
            location.time_of_day.sync_to_realtime();
        }
        if ui
            .selectable_label(location.time_of_day.mode == TimeMode::Override, "Manual")
            .clicked()
            && location.time_of_day.mode != TimeMode::Override
        {
            // Switch to override mode, keeping current time.
            let current_speed = location.time_of_day.speed_multiplier;
            location.time_of_day.mode = TimeMode::Override;
            location.time_of_day.set_speed(current_speed);
        }
    });

    // Date displayed alongside local time, so the calendar date matches
    // what a clock at the camera's longitude would say (UTC and local
    // disagree on the date for ~half of any given day).
    let local_date = location.time_of_day.current_date_at_longitude(lon_deg);
    let (utc_h, utc_m, utc_s) = seconds_to_hms(location.time_of_day.current_utc_seconds());
    ui.label(format!(
        "Date: {}-{:02}-{:02}",
        local_date.year, local_date.month, local_date.day
    ));
    ui.label(format!("UTC: {utc_h:02}:{utc_m:02}:{utc_s:02}"));

    // Display current local time with timezone offset.
    let local_hours = location.time_of_day.local_hours_at_longitude(lon_deg);
    let offset_hours = lon_deg / 15.0;
    let (hours, minutes, seconds) = seconds_to_hms(local_hours * SECONDS_PER_HOUR);
    let offset_sign = if offset_hours >= 0.0 { "+" } else { "" };

    let is_override = location.time_of_day.mode == TimeMode::Override;
    ui.horizontal(|ui| {
        ui.label(format!(
            "Local: {hours:02}:{minutes:02}:{seconds:02} (UTC{offset_sign}{offset_hours:.1})"
        ));
        if is_override {
            let mut slider_hours = local_hours;
            ui.add(
                egui::Slider::new(&mut slider_hours, 0.0..=24.0)
                    .text("hours")
                    .fixed_decimals(2),
            );
            // Only update time if there was a significant change.
            if (slider_hours - local_hours).abs() > 0.01 {
                // Local-time input → UTC via the explicit inverse
                // projection. Preserves the local date by going
                // through (slider_hours, current local date) rather
                // than (slider_hours, UTC date).
                let (utc_seconds, utc_date) =
                    local_to_utc(slider_hours * SECONDS_PER_HOUR, local_date, lon_deg);
                location.time_of_day.set_override_utc(utc_date, utc_seconds);
            }
        }
    });

    // Time and date controls (only in override mode).
    if is_override {
        // Local-date picker. Setting a new date preserves the local
        // hour at this longitude — the date change goes through
        // `local_to_utc` so cross-midnight cases are handled (UTC
        // may need to be the day before/after the picked local
        // date).
        ui.horizontal(|ui| {
            ui.label("Date:");
            if let Some(mut picked) = local_date.to_naive() {
                let before = picked;
                ui.add(egui_extras::DatePickerButton::new(&mut picked).id_salt("local_date"));
                if picked != before {
                    let new_local_date = crate::world::time_of_day::SimpleDate::from_naive(picked);
                    let (utc_seconds, utc_date) = crate::world::time_of_day::local_to_utc(
                        local_hours * SECONDS_PER_HOUR,
                        new_local_date,
                        lon_deg,
                    );
                    location.time_of_day.set_override_utc(utc_date, utc_seconds);
                }
            } else {
                ui.label("(invalid date)");
            }
        });

        // Show sun declination for reference.
        let declination = location.time_of_day.sun_declination_deg();
        ui.label(format!("Sun declination: {declination:.1}\u{00b0}"));
    }

    // Moon state — useful for verifying night-side lighting and phase logic.
    let moon = compute_moon_state(&location.time_of_day);
    let local_up = position.normalize().as_vec3();
    let moon_altitude_deg = moon.altitude_at(local_up).to_degrees();
    let visible = if moon_altitude_deg > 0.0 {
        "up"
    } else {
        "down"
    };
    ui.label(format!(
        "Moon: {phase} ({pct:.0}%), altitude {alt:.1}\u{00b0} ({visible})",
        phase = moon.phase_name(),
        pct = moon.illuminated_fraction * 100.0,
        alt = moon_altitude_deg,
    ));

    // Time-speed controls — pause toggle + logarithmic slider from
    // 0.1× to 100 000×. Pause is a separate boolean so the slider
    // remembers the user's previous non-zero speed across un-pause.
    ui.horizontal(|ui| {
        ui.label("Time speed:");
        let current_speed = location.time_of_day.speed_multiplier;
        let is_paused = current_speed == 0.0;
        if ui.selectable_label(is_paused, "Pause").clicked() {
            if is_paused {
                let resume = location.time_of_day.last_unpaused_speed.max(0.1);
                location.time_of_day.set_speed(resume);
            } else {
                location.time_of_day.last_unpaused_speed = current_speed;
                location.time_of_day.set_speed(0.0);
            }
        }
        ui.add_enabled_ui(!is_paused, |ui| {
            let mut speed = if is_paused {
                location.time_of_day.last_unpaused_speed.max(0.1)
            } else {
                current_speed
            };
            if ui
                .add(
                    egui::Slider::new(&mut speed, 0.1_f32..=100_000.0_f32)
                        .logarithmic(true)
                        .text("×"),
                )
                .changed()
                && !is_paused
            {
                location.time_of_day.set_speed(speed);
                if location.time_of_day.mode == TimeMode::Realtime
                    && (speed - 1.0).abs() > f32::EPSILON
                {
                    location.time_of_day.mode = TimeMode::Override;
                }
            }
        });
    });

    // Execute geocoding/teleport actions.
    if start_geocoding {
        location.geocoding_state.start_request(
            current_time,
            &location.http_client,
            &location.spawner,
        );
    }

    if start_reverse_geocoding {
        location.geocoding_state.start_reverse_request(
            lat_deg,
            lon_deg,
            current_time,
            &location.http_client,
            &location.spawner,
        );
    }

    if let Some((lat, lon)) = new_coords {
        location.geocoding_state.results.clear();
        location
            .teleport_state
            .request(lat, lon, &location.http_client, &location.spawner);
    }
}

/// Render the compass row: a small painted rose showing the camera's
/// current heading, a numeric / cardinal readout, a 0–360° slider, and
/// quick-snap buttons for the four cardinal directions.
///
/// The flight-camera query is read-only here. Heading changes are queued
/// through [`HeadingRequest`] so the camera-control system owns the
/// `&mut FlightCamera` borrow — avoiding a query-overlap with the camera
/// tab's read of the same component within this UI system.
fn render_compass(
    ui: &mut egui::Ui,
    heading_request: &mut HeadingRequest,
    flight_camera_query: &Query<&FlightCamera>,
    position: DVec3,
) {
    let Ok(flight_cam) = flight_camera_query.single() else {
        return;
    };

    // Local tangent basis at the camera position. Matches the bake /
    // shadow-uniform math (see `cloud_shadow_bake.wgsl` and
    // `bevy_pbr_clouds_planet::resources`): `world_north` projected
    // onto the tangent plane, falling back to `world_east` near the
    // poles where the projection is degenerate.
    let up = position.normalize().as_vec3();
    let world_north = Vec3::Z;
    let mut local_north = (world_north - up * world_north.dot(up)).normalize_or_zero();
    if local_north.length_squared() < 0.5 {
        let world_east = Vec3::X;
        local_north = (world_east - up * world_east.dot(up)).normalize_or_zero();
    }
    // `local_north.cross(up)` is geographic east (+Y at lon=0, equator):
    // `up.cross(local_north)` would give -Y (west) and silently flip the
    // compass labels. Keep this consistent with `process_heading_request`.
    let local_east = local_north.cross(up).normalize_or_zero();

    // Current bearing (clockwise from north, in [0, 360)).
    let horizontal = flight_cam.direction - up * flight_cam.direction.dot(up);
    let bearing_deg = if horizontal.length_squared() < 1e-8 {
        0.0
    } else {
        let raw = local_east
            .dot(horizontal)
            .atan2(local_north.dot(horizontal))
            .to_degrees();
        if raw < 0.0 { raw + 360.0 } else { raw }
    };
    let cardinal = cardinal_for_bearing(bearing_deg);

    ui.separator();

    ui.horizontal(|ui| {
        // Painted compass rose: small circle, N marker at top, current-
        // heading arrow.
        let size = 56.0;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
        let painter = ui.painter();
        let center = rect.center();
        let radius = size * 0.45;
        let stroke_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
        painter.circle_stroke(center, radius, egui::Stroke::new(1.0, stroke_color));
        // Cardinal labels around the rose.
        for (label, deg) in [("N", 0.0_f32), ("E", 90.0), ("S", 180.0), ("W", 270.0)] {
            let rad = deg.to_radians();
            // egui Y is screen-down, so subtract the cos term to put
            // north at the top of the rose.
            let pos = center + egui::vec2(rad.sin() * radius, -rad.cos() * radius);
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(10.0),
                stroke_color,
            );
        }
        // Heading arrow.
        let rad = bearing_deg.to_radians();
        let arrow_end = center + egui::vec2(rad.sin() * radius * 0.8, -rad.cos() * radius * 0.8);
        painter.line_segment(
            [center, arrow_end],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 80, 80)),
        );
        painter.circle_filled(arrow_end, 2.5, egui::Color32::from_rgb(255, 80, 80));

        ui.vertical(|ui| {
            ui.label(format!("Heading: {bearing_deg:5.1}\u{00b0} ({cardinal})"));
            let mut new_bearing = bearing_deg;
            if ui
                .add(
                    egui::Slider::new(&mut new_bearing, 0.0..=360.0)
                        .suffix("\u{00b0}")
                        .smart_aim(false),
                )
                .changed()
            {
                heading_request.request(new_bearing);
            }
            ui.horizontal(|ui| {
                if ui.button("N").clicked() {
                    heading_request.request(0.0);
                }
                if ui.button("E").clicked() {
                    heading_request.request(90.0);
                }
                if ui.button("S").clicked() {
                    heading_request.request(180.0);
                }
                if ui.button("W").clicked() {
                    heading_request.request(270.0);
                }
            });
        });
    });
}

/// Render the precise-translation controls: a distance selector plus
/// N/E/S/W buttons that move the camera an exact great-circle distance
/// along that bearing. Edits route through [`TranslateRequest`] so the
/// camera-control system owns the position write.
fn render_translation_controls(
    ui: &mut egui::Ui,
    distance_m: &mut f64,
    translate_request: &mut TranslateRequest,
) {
    ui.separator();
    ui.label("Precise move:");
    ui.horizontal(|ui| {
        ui.label("Distance:");
        for (label, metres) in [
            ("100 m", 100.0),
            ("1 km", 1000.0),
            ("10 km", 10_000.0),
            ("50 km", 50_000.0),
            ("100 km", 100_000.0),
        ] {
            // Bit-compare is fine here: the stored value only ever comes
            // from these exact literals.
            if ui
                .selectable_label((*distance_m - metres).abs() < f64::EPSILON, label)
                .clicked()
            {
                *distance_m = metres;
            }
        }
    });
    ui.horizontal(|ui| {
        let d = *distance_m;
        ui.label(format!("Move {:.0} m:", d));
        if ui.button("N").clicked() {
            translate_request.request(0.0, d);
        }
        if ui.button("E").clicked() {
            translate_request.request(90.0, d);
        }
        if ui.button("S").clicked() {
            translate_request.request(180.0, d);
        }
        if ui.button("W").clicked() {
            translate_request.request(270.0, d);
        }
    });
}

/// 16-point cardinal label for a bearing in degrees (clockwise from north).
fn cardinal_for_bearing(deg: f32) -> &'static str {
    const LABELS: [&str; 16] = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW",
        "NW", "NNW",
    ];
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = (((deg / 22.5) + 0.5).floor() as i32).rem_euclid(16) as usize;
    LABELS[idx]
}
