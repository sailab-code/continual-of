// =============================================================================================================
// ===================================== SYNC-CONTROL-RELATED AJAX CALLS =======================================
// =============================================================================================================

// AJAX: tell the system that it is free to run, the GUI is not following it
function vprocessor_allow_processing() {
    log("Requested: vprocessor_allow_processing");
    block_all_GUI(true);
    $.ajax({
        type: "GET",
        url: "/vprocessor_allow_processing",
        success: function (byte_str) {
            log("Returned: vprocessor_allow_processing (data=" + byte_str + ")");
            block_all_GUI(false);
        },
        error: function () {
            log("Returned: vprocessor_allow_processing (error)");
            block_all_GUI(false);
            handle_ajax_error(function () {
                alert("Error while telling the system to be free!");
            });
        }
    });
}

// AJAX: tell the system that it should process a new frame (then we get the data and draw them)
// warning: the system might say "hey, I want to ask for a supervision, so I did not process any new frame"
function vprocessor_allow_processing_next_frame_only() {
    log("Requested: vprocessor_allow_processing_next_frame_only");
    _ajax_pending_requests++; // fake - added for safety
    block_all_GUI(true);
    $.ajax({
        type: "GET",
        url: "/vprocessor_allow_processing_next_frame_only",
        success: function (byte_str) {

            // the system returns the frame number it is processing right now (or 0 if it is not enabled)
            log("Returned: vprocessor_allow_processing_next_frame_only (data=" + byte_str + ")");

            if (byte_str === "0") {
                stop_following();
            } else {

                // recall that we are in the middle of the play-thread operation
                // so we have to get and draw the next frame
                if (_sync_follow_last_frame_text !== byte_str) { // ensure the system has moved to a new frame
                    var frame_number = byte_str;

                    // TODO temporarily disabled (replaced by the code right after this comment block)
                    /*
                    if (frame_number[0] != "-") { // ensure that the system is not asking for a supervision
                        set_currently_displayed_frame_number(frame_number);
                        set_currently_displayed_max_frame_number(frame_number);
                        _sync_follow_last_frame_text = frame_number;
                        do_action("next", true); // get data and draw!
                    } else {
                        set_currently_displayed_frame_number(frame_number.substring(1)) // removing the "-"
                        set_currently_displayed_max_frame_number(frame_number.substring(1));
                        _sync_follow_last_frame_text = frame_number.substring(1);
                        if (_sync_last_system_initiative_frame_text == _sync_follow_last_frame_text) {
                            send_supervision('<skip>');
                        } else {
                            _sync_last_system_initiative_frame_text = _sync_follow_last_frame_text;
                            system_initiative();
                        }
                    }
                    */

                    set_currently_displayed_frame_number(frame_number);
                    set_currently_displayed_max_frame_number(frame_number);
                    _sync_follow_last_frame_text = frame_number;
                    do_action("next", true); // get data and draw!
                }
            }

            // fake (it was incremented when starting the play-thread operations, for safety)
            _ajax_pending_requests--;
            block_all_GUI(false);
        },
        error: function () {
            log("Returned: vprocessor_allow_processing_next_frame_only (error)");
            block_all_GUI(false);
            handle_ajax_error(function () {
                alert("Error while telling the system to process next frame!");
            });
        }
    });
}

// AJAX: tell the system to stop as soon as possible (i.e., when it finished processing the current frame)
function vprocessor_disable_processing_asap() {
    log("Requested: vprocessor_disable_processing_asap");
    block_all_GUI(true);
    $.ajax({
        type: "GET",
        url: "/vprocessor_disable_processing_asap",
        success: function (byte_str) {
            log("Returned: vprocessor_disable_processing_asap (data=" + byte_str + ")");
            block_all_GUI(false);
        },
        error: function () {
            log("Returned: vprocessor_disable_processing_asap (error)");
            block_all_GUI(false);
            handle_ajax_error(function () {
                alert("Error while telling the system to stop ASAP!");
            });
        }
    });
}

// =============================================================================================================
// ================================= GET/SEND-INFO-RELATED AJAX CALLS ==========================================
// =============================================================================================================

// AJAX: send a new value for an option
function send_option_change(name, value_string, option_element_html) {
    log("Requested: vprocessor_change_option (name=" + name + " value=" + value_string + ")");
    if (name.trim().length === 0) {
        alert("Missing option name!");
        return;
    }
    if (value_string.trim().length === 0) {
        alert("Missing option value!");
        return;
    }
    $.ajax({
        type: "GET",
        url: "/vprocessor_change_option",
        data: {
            opt_name: name,
            opt_value: value_string
        },
        success: function (byte_str) {
            log("Returned: vprocessor_change_option (data=" + byte_str + ")");
            if (byte_str === "1") {
                option_element_html.val(value_string);
            } else {
                $('#text_option').val(option_element_html.val());
            }
        },
        error: function () {
            log("Returned: vprocessor_change_option (error)");
            handle_ajax_error(function () {
                alert("Error while sending option change! " +
                    "name=" + name + " value=" + value_string);
            });
        }
    });
}

// AJAX: check if the system is in pause mode etc... and apply some changes due to such check
function get_and_react_to_sync_status(on_load) {
    log("Requested: vprocessor_is_processing_allowed");
    $.ajax({
        type: "GET",
        url: "/vprocessor_is_processing_allowed",
        success: function (byte_str) {
            log("Returned: vprocessor_is_processing_allowed (data=" + byte_str + ")");
            if (byte_str === "0") {
                sync_pause_on();
            }
        },
        error: function () {
            log("Returned: vprocessor_is_processing_allowed (error)");
            if (!on_load) {
                handle_ajax_error(function () {
                    alert("Error while asking for the status of the system!");
                });
            } else {
                $('#button_sync_on').prop("disabled", true);
                $('#button_sync_pause').prop("disabled", true);
            }
        }
    });
}

// AJAX: getting system options (always from file)
function get_options(on_load) {
    log("Requested: options");
    $.ajax({
        type: "GET",
        url: "/options",
        success: function (byte_str) {
            log("Returned: options");

            var options = JSON.parse(byte_str);

            _w = options.w;
            _h = options.h;
            set_currently_displayed_fps(options.fps);

            var layer = $('#text_layer_id').val();
            if (options.worker.net.hasOwnProperty('prob_layers_sizes')) {
                if (options.worker.net.prob_layers_sizes === undefined
                    || options.worker.net.prob_layers_sizes === null
                    ||options.worker.net.prob_layers_sizes.length === 0) {
                    _features_per_layer = options.worker.net.m;
                } else {
                    _features_per_layer = [];
                    let z = 0;
                    for (z = 0; z < options.worker.net.prob_layers_sizes.length; z++) {
                        let s = options.worker.net.prob_layers_sizes[z];
                        _features_per_layer[z] =
                            (s.length > 0 && s[s.length-1] > 0) ? s[s.length-1] : options.worker.net.m[z];
                    }
                }
                _features = _features_per_layer[layer];
                $('#text_layers').val(options.worker.net.fe_layers + options.worker.net.sem_layers - 1);
                $('#text_features_range').val("0-" + (_features - 1));
            } else {
                if (options.worker.net.hasOwnProperty('m')) {
                    _features_per_layer = options.worker.net.m;
                    _features = _features_per_layer[layer];
                    $('#text_layers').val(options.worker.net.fe_layers - 1);
                    $('#text_features_range').val("0-" + (_features - 1));
                }
            }

            if (options.worker.net.hasOwnProperty('num_what')) {
                _num_what = options.worker.net.num_what;
                $('#text_what_range').val("0-" + (_num_what - 1));
            } else {
                _num_what = 0;
            }

            if (options.worker.net.hasOwnProperty('num_where')) {
                _num_where = options.worker.net.num_where;
                $('#text_where_range').val("0-" + (_num_where - 1));
            } else {
                _num_where = 0;
            }

            _outputs = options.output_folder_data_types;

            // generic outputs that could be shown
            _generic_outputs = {}
            _text_range_generic_outputs = {}
            _normalize_generic_outputs = {}
            let first_output = true;
            for (var output_name in _outputs) {
                var output_type = _outputs[output_name];
                if (output_type === 'IMAGE' || output_type == 'BINARY') {
                    if (output_name === 'frames') continue;
                    if (output_name === 'motion') continue;
                    if (output_name === 'sup-probs') continue;
                    if (output_name === 'sup.indices') continue;
                    if (output_name === 'sup.targets') continue;
                    $('#output_list').append($('<option>', {
                        value: output_type,
                        text: output_name
                    }))
                    if (output_type === 'IMAGE') {
                        _text_range_generic_outputs[output_name] = "none";
                        _normalize_generic_outputs[output_name] = false;
                    }
                    if (output_type === 'BINARY') {
                        _text_range_generic_outputs[output_name] = "0-last";
                        if (!output_name.includes("prob")) {
                            _normalize_generic_outputs[output_name] = true;
                        } else {
                            _normalize_generic_outputs[output_name] = false;
                        }
                    }
                    if (first_output) {
                        first_output = false;
                        $('#output_list').change();
                    }
                }
            }

            // encoding each option using the keys needed to reach it, such as name1.name2.name3 etc...
            var queue = [];
            var names = [];
            var values = [];
            for (var name in options.worker) {
                if (options.worker.hasOwnProperty(name))
                    queue.push({"element": options.worker[name], "fullname": name});
            }

            while (queue.length > 0) {
                var q = queue.pop();
                var opt = q.element;
                var full_name = q.fullname;
                if (typeof opt !== 'object') {
                    names.push(full_name);
                    values.push(opt);
                } else {
                    for (var n in opt) {
                        if (opt.hasOwnProperty(n))
                            queue.push({"element": opt[n], "fullname": full_name + "." + n});
                    }
                }
            }

            // options that could be updated
            for (var i = 0; i < names.length; i = i + 1) {
                $('#option_list').append($('<option>', {
                    value: values[i],
                    text: names[i]
                }));
            }
            $('#text_option').val($('#option_list').val());

            _predictors = options.worker.net.supervised_categories;

            if (_predictors > 0)
                $('#text_predictions_range').val("0-" + (_predictors - 1));
            else
                $('#text_predictions_range').val("none");

            if (on_load) {
                template_predictions();
            }
        },
        error: function () {
            log("Returned: options (error)");
            handle_ajax_error(function () {
                alert("Error while getting system options!");
            });
        }
    });
}

// AJAX: get last frame number
function get_last_frame_number() {
    log("Requested: last_frame_number");
    $.ajax({
        type: "GET",
        url: "/last_frame_number",
        success: function (byte_str) {
            log("Returned: last_frame_number");

            var last_frame = parseInt(byte_str);
            set_currently_displayed_max_frame_number(last_frame);
        },
        error: function () {
            log("Returned: last_frame_number (error)");
            handle_ajax_error(function () {
                alert("Error while getting last frame number!");
            });
        }
    });
}

// AJAX: send command
function send_command(name, value_string, custom_callback) {
    log("Requested: vprocessor_command (" + name + ")");
    $.ajax({
        type: "GET",
        url: "/vprocessor_command",
        data: {
            "opt_name": name,
            "opt_value": value_string
        },
        success: function (byte_str) {
            log("Returned: vprocessor_command (" + name + ") (data=" + byte_str + ")");
            if (custom_callback !== undefined) {
                custom_callback();
            }
        },
        error: function () {
            log("Returned: vprocessor_command (" + name + ") (error)");
            handle_ajax_error(function () {
                alert("Error while sending command! (" + name + ")");
            });
            if (custom_callback !== undefined) {
                custom_callback();
            }
        }
    });
}
