//! Gameplay tab for the debug UI.
//!
//! Provides vehicle spawning and status display.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_egui::egui;

use crate::camera::CameraModeState;
use crate::vehicle::{Vehicle, VehicleActions, VehicleDefinitions, VehicleState};

/// Resources for the gameplay tab.
#[derive(SystemParam)]
pub(super) struct GameplayParams<'w, 's> {
    pub camera_mode: Res<'w, CameraModeState>,
    pub vehicle_definitions: Res<'w, VehicleDefinitions>,
    pub vehicle_actions: ResMut<'w, VehicleActions>,
    pub vehicle_query: Query<'w, 's, (&'static Vehicle, &'static VehicleState)>,
}

/// Render the gameplay tab content.
pub(super) fn render_gameplay_tab(ui: &mut egui::Ui, gameplay: &mut GameplayParams) {
    // Vehicle controls.
    ui.label("Vehicles:");
    if gameplay.vehicle_definitions.vehicles.is_empty() {
        ui.label("Loading...");
    } else {
        ui.horizontal_wrapped(|ui| {
            for (idx, def) in gameplay.vehicle_definitions.vehicles.iter().enumerate() {
                if ui
                    .button(&def.name)
                    .on_hover_text(&def.description)
                    .clicked()
                {
                    gameplay.vehicle_actions.request_spawn(idx);
                }
            }
        });
    }

    // Show vehicle stats when in FollowEntity mode.
    if gameplay.camera_mode.is_follow_entity()
        && let Some((vehicle, state)) = gameplay.vehicle_query.iter().next()
    {
        ui.separator();
        ui.label(format!("Vehicle: {}", vehicle.name));
        ui.label(format!("Speed: {:.0} km/h", state.speed * 3.6));
        ui.label(if state.grounded {
            "Status: Grounded"
        } else {
            "Status: Airborne"
        });

        if ui.button("Exit vehicle (E)").clicked() {
            gameplay.vehicle_actions.request_exit();
        }
    }
}
