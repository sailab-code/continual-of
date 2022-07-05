// status variables
var _w = 0; // width of the currently considered video
var _h = 0; // height of the currently considered video
var _pending_ajax_calls = []; // list of the currently issued ajax calls
var _flag_error = false; // error status flag
var _play_thread; // thread triggered by a timer for issuing a show-next-frame request
var _log_to_console = true; // flag to indicate whether we want to print something to the console
var _outputs; // dictionary with the output elements that are avaialbe
var _selected_output; // currently selected generic output to show/hide
var _generic_outputs; // status (show/hide) of generic outputs
var _normalize_generic_outputs; // normalize or not binary data
var _text_range_generic_outputs; // range of channels for binary data

var _views_row = 0; // currently available row in the view-table
var _views_col = 0; // currently available column in the view-table
var _views_max_cols = 2; // option
var _views_caption_2_canvas_context = []; // map from view-caption to the corresponding canvas context
var _ajax_pending_requests = 0; // counter of the canvas that are currently waiting to be drawn
var _ajax_requests_that_did_not_failed = 0; // counter of canvas not-drawn due to errors
var _views_visualization_changed = false; // flag to indicate that canvas must be recreated before drawing

var _sync_follow = false; // sync mode: flag to indicate that the GUI is driving the system (following)
var _sync_follow_last_frame_text = ""; // frame number (string) of the last frame received in "following" mode
var _sync_paused = false; // sync mode: flag to indicate that the system is paused (it is physically paused!)

var _frame_pixels; // 3-channel representation of the frame pixels (needed by multiple views)
var _motion; // array of floats with the motion field (needed by multiple routines)
var _motion_c_first; // flag to describe the motion data

var _foa; // focus of attention array (4 components)
var _foa_registered_mouse = false; // flag that indicates that mouse events have been already registered

var _block_gui_requests = 0; // number of requests to lock all the GUI

var _predictors = 0; // max number of predictors
var _predictions_range_text = ''; // range of predictor-related maps displayed in the GUI (example: "0-5")
var _predictions; // predictions (needed by multiple views)
var _predictions_c_first; // flag to describe the prediction data

var _interactions_draw_style; // draw style ("max", "top3", ...)
var _interactions_backup_pixels; // backup of the interaction-view before drawing markers (crosses, ...)
var _interactions_finished_drawing = true; // flag that indicates that the drawing-call has ended
var _interactions_hidden_canvas_html; // hidden canvas HTML (scaling operations)
var _interactions_hidden_canvas; // hidden canvas (scaling operations)
var _interactions_image_obj = new Image(); // used for scaling operations
var _interactions_scale = 1.0; // scale
var _interactions_drag_start_xy; // supervision-related coordinates
var _save_interaction_pngs = false; // it will save each interaction window to a PNG file

var _sup_label_map = {}; // map from textual label to predictor ID
var _sup_label_map_inv = {}; // map from predictor ID to textual label
var _sup_label_count = {}; // map from textual label to supervision counts
var _sup_x, _sup_y, _sup_w, _sup_h; // coordinates of the supervision we are going to send

var _sup_targets; // supervisions that were provided to the system
var _sup_indices; // supervisions that were provided to the system

var _colors = generate_array_of_colors(); // these are the colors used in some views

// TODO temporarily disabled
/*
var _sync_last_system_initiative_frame_text = ""; // frame number (string) where initiative was taken
*/

// =============================================================================================================
// ====================================== GUI-HANDLING-RELATED FUNCTIONS =======================================
// =============================================================================================================

// log to console
function log(text) {
    if (_log_to_console) {
        console.log(text);
    }
}

// get currently displayed frame number
function get_currently_displayed_frame_number() {
    return parseInt($('#text_cur_frame').val());
}

// get currently displayed "max" frame number
function get_currently_displayed_max_frame_number() {
    return parseInt($('#text_frames').val());
}

// get currently displayed fps
function get_currently_displayed_fps() {
    return parseInt($('#text_fps').val());
}

// guess what?
function set_currently_displayed_frame_number(num) {
    $('#text_cur_frame').val(num);
}

// guess what?
function set_currently_displayed_max_frame_number(num) {
    $('#text_frames').val(num);
}

// guess what?
function set_currently_displayed_fps(num) {
    $('#text_fps').val(num);
}

// get the GUI options and pack them into a dictionary
function get_GUI_options() {
    var fps = get_currently_displayed_fps();
    var frame = get_currently_displayed_frame_number();
    var tot_frames = get_currently_displayed_max_frame_number();
    var showVideo = $('#checkbox_video').is(':checked');
    var showMotion = $('#checkbox_motion').is(':checked');
    var showPredictedMotion = $('#checkbox_predicted_motion').is(':checked');
    var showPredictedBackwardMotion = $('#checkbox_predicted_backward_motion').is(':checked');
    var showIHSMotion = $('#checkbox_ihs_motion').is(':checked');
    var showFocus = $('#checkbox_focus').is(':checked');
    var showDetails = $('#checkbox_details').is(':checked');
    var motionMap = $('#radio_motion_map').is(':checked');
    var showPredictions = $('#checkbox_predictions').is(':checked');
    var showInteractions = $('#checkbox_interaction').is(':checked');
    var showSupervisions = $('#checkbox_supervisions').is(':checked');
    var normalizePredictions = $('#checkbox_predictions_normalization').is(':checked');

    var interactionsDrawStyle = "none";
    if ($('#radio_predictions_max').is(':checked')) {
        interactionsDrawStyle = "max";
    }
    if ($('#radio_predictions_top').is(':checked')) {
        interactionsDrawStyle = "top3";
    }
    if ($('#radio_predictions_focus').is(':checked')) {
        interactionsDrawStyle = "focus_seg";
    }

    _interactions_draw_style = interactionsDrawStyle;

    var motionDrawStyle = "map";
    if (!motionMap) {
        motionDrawStyle = "lines";
    }

    return {
        fps: fps,
        frame: frame,
        tot_frames: tot_frames,
        showVideo: showVideo,
        showMotion: showMotion,
        showPredictedMotion: showPredictedMotion,
        showPredictedBackwardMotion: showPredictedBackwardMotion,
        showIHSMotion: showIHSMotion,
        showFocus: showFocus,
        showDetails: showDetails,
        motionMap: motionMap,
        motionDrawStyle: motionDrawStyle,
        showPredictions: showPredictions,
        showInteractions: showInteractions,
        showSupervisions: showSupervisions,
        normalizePredictions: normalizePredictions,
        showGenericOutputs: _generic_outputs,
        normalizeGenericOutputs: _normalize_generic_outputs,
        textRangeGenericOutputs: _text_range_generic_outputs,
    };
}

// add a new canvas to the view-table or return an already created one
function get_canvas(caption, w, h) {
    var ctx = _views_caption_2_canvas_context[caption];
    var canvas_id;
    var _views_last_td_id;

    if (ctx === undefined) {
        if (_views_col === _views_max_cols || _views_col === 0) {
            _views_row = _views_row + 1;
            _views_col = 1;
            canvas_id = "canvas" + ((_views_row - 1) * _views_max_cols + _views_col);
            var line = '<tr><td><canvas width="100%" height="100%" ' +
                'oncontextmenu="return false;" id="' + canvas_id + '"/><br/><p id="' + caption + '">' + caption +
                '</p></td>';
            for (var q = 1; q < _views_max_cols; q = q + 1) {
                _views_last_td_id = "td" + ((_views_row - 1) * _views_max_cols + q);
                line = line + '<td id="' + _views_last_td_id + '"></td>';
            }
            line = line + '</tr>';
            $('#main_table').append(line);
        } else {
            _views_col = _views_col + 1;
            canvas_id = "canvas" + ((_views_row - 1) * _views_max_cols + _views_col);
            _views_last_td_id = "td" + ((_views_row - 1) * _views_max_cols + _views_col - 1);
            $('#' + _views_last_td_id).html('<canvas width="100%" height="100%" ' +
                'oncontextmenu="return false;" id="' + canvas_id + '"/><br/><p id="' + caption + '">' + caption +
                '</p>');
        }

        var c = document.getElementById(canvas_id);
        ctx = c.getContext('2d');
        ctx.canvas.width = w;
        ctx.canvas.height = h;

        _views_caption_2_canvas_context[caption] = ctx;

        if (caption === "Interaction") {
            _interactions_hidden_canvas_html = document.createElement('canvas'); // hidden canvas
            _interactions_hidden_canvas = _interactions_hidden_canvas_html.getContext('2d');
            _interactions_hidden_canvas.canvas.width = w;
            _interactions_hidden_canvas.canvas.height = h;

            add_mouse_listeners_to_interaction(ctx);
        }

        return ctx;
    } else {
        return ctx;
    }
}

// rename predictor-related canvas
function rename_predictor_related_canvas() {
    if (_sup_label_map_inv !== undefined) {
        let k;
        let caption;
        for (k = 0; k < _predictors; k++) {
            caption = "Predictor " + k;
            if (is_canvas_existing(caption) && _sup_label_map_inv[k] !== undefined) {
                document.getElementById(caption).innerHTML = caption + " (" + _sup_label_map_inv[k] + ")";
            }
        }
    }
}

// show a generic output
function show_generic_output(output_name, output_type) {
    _generic_outputs[output_name] = output_type
    _views_visualization_changed = true;
}

// hide a generic output
function hide_generic_output(output_name) {
    if (_generic_outputs.hasOwnProperty(output_name)) {
        delete _generic_outputs[output_name];
        _views_visualization_changed = true;
    }
}

// destroy all canvas and all the pending ajax calls, reset the canvas status
function reset_all_canvas() {
    reset_drawing_related_routines();
    _views_caption_2_canvas_context = [];
    _views_row = 0;
    _views_col = 0;
    document.getElementById('main_table').innerHTML = "<tbody></tbody>";
    document.getElementById("json").innerHTML = "";
    hide_label_table(true);
}

// reset the drawing-related routines
function reset_drawing_related_routines() {
    clear_ajax_calls();
    _ajax_pending_requests = 0;
    _ajax_requests_that_did_not_failed = 0;
}

// disable or show some GUI controls (mostly checkboxes)
function disable_GUI_options_about_what_to_show(disable_or_not) {
    if (disable_or_not) {
        $('#checkbox_video').attr('disabled', true);
        $('#checkbox_motion').attr('disabled', true);
        $('#checkbox_details').attr('disabled', true);
        $('#checkbox_predictions').attr('disabled', true);
        $('#checkbox_interaction').attr('disabled', true);
        $('#checkbox_supervisions').attr('disabled', true);
    } else {
        $('#checkbox_video').removeAttr('disabled');
        $('#checkbox_motion').removeAttr('disabled');
        $('#checkbox_details').removeAttr('disabled');
        $('#checkbox_predictions').removeAttr('disabled');
        $('#checkbox_interaction').removeAttr('disabled');
        $('#checkbox_supervisions').removeAttr('disabled');
    }
}

// disable or show some GUI controls (mostly checkboxes) given the available output elements
function disable_unsupported_GUI_options() {
    let stats = false;
    let output_types = Object.keys(_outputs);
    for (let i = 0; i < output_types.length; i = i + 1) {
        let k = output_types[i];
        if (k.startsWith("stats")) {
            stats = true;
        }
    }
    $('#checkbox_video').attr('disabled', false);
    if (!_outputs.hasOwnProperty("motion")) {
        if ($('#checkbox_motion').is(':checked')) {
            $('#checkbox_motion').attr('disabled', false);
            $('#checkbox_motion').click();
        }
        $('#checkbox_motion').attr('disabled', true);
    }
    if (!stats) {
        if ($('#checkbox_details').is(':checked')) {
            $('#checkbox_details').attr('disabled', false);
            $('#checkbox_details').click();
        }
        $('#checkbox_details').attr('disabled', true);
    }
    if (!_outputs.hasOwnProperty("sup-probs")) {
        if ($('#checkbox_predictions').is(':checked')) {
            $('#checkbox_predictions').attr('disabled', false);
            $('#checkbox_predictions').click();
        }
        $('#checkbox_predictions').attr('disabled', true);
        if ($('#checkbox_interaction').is(':checked')) {
            $('#checkbox_interaction').attr('disabled', false);
            $('#checkbox_interaction').click();
        }
        $('#checkbox_interaction').attr('disabled', true);
        if ($('#checkbox_supervisions').is(':checked')) {
            $('#checkbox_supervisions').attr('disabled', false);
            $('#checkbox_supervisions').click();
        }
        $('#checkbox_supervisions').attr('disabled', true);
    }
    if (!_outputs.hasOwnProperty("stats.worker")) {
        if ($('#checkbox_focus').attr(':checked')) {
            $('#checkbox_focus').attr('disabled', false);
            $('#checkbox_focus').click();
        }
        $('#checkbox_focus').attr('disabled', true);
    }
}

// handle the generic ajax error
function handle_ajax_error(custom_handler) {
    if (!_flag_error) {
        _flag_error = true;
        _views_visualization_changed = true;
        do_action("stop");
        custom_handler();
    }
}

// handler to generic ajax error happened in drawing-related calls
function handle_ajax_draw_error(ctx, get_only, update_counters) {
    if (!get_only) {
        drawNotFoundOrError(ctx, ctx.canvas.width, ctx.canvas.height);
    }
    if (update_counters) {
        _ajax_pending_requests--;
        _ajax_requests_that_did_not_failed--;

        if (_ajax_requests_that_did_not_failed === 0) {
            handle_ajax_error(function () {
                    log("All drawing operations failed, stopping");
                    alert("All drawing operations failed, stopping!");
            });
        } else {
            log("Drawing operations that are/can still be served with success=" +
                _ajax_requests_that_did_not_failed);
        }
    }
}

// clear all the pending ajax calls
function clear_ajax_calls() {
    for (var i = 0; i < _pending_ajax_calls.length; i++) {
        _pending_ajax_calls[i].abort();
    }
    _pending_ajax_calls = [];
    _ajax_requests_that_did_not_failed = 0;
}

function block_all_GUI(block_yes) {
    if (block_yes) {
        document.getElementById('disabling_div').style.display = 'block';
        _block_gui_requests = _block_gui_requests + 1;
    } else {
        _block_gui_requests = _block_gui_requests - 1;
        if (_block_gui_requests === 0) {
            document.getElementById('disabling_div').style.display = 'none';
        }
    }
}

// add mouse listeners to interaction window
function add_mouse_listeners_to_interaction(ctx) {

    // handler of all mouse-related interactions with this view
    var mouse_event_handler = function (event) {

        // only catch events when in sync (following) mode
        if (_sync_follow) {

            // clicking the mouse button pauses the system
            if (!_sync_paused) {
                if (event.type === "mousedown") {
                    if (event.which !== 3) {
                        // left (well, actually not-right) click
                        sync_pause_on();
                    }
                }
                return;
            } else {

                // do not catch events if the system is drawing something in the interaction window
                if (!_interactions_finished_drawing) {
                    return;
                }

                // do not catch mouse movements if a drag operation was not started before
                if (event.type === "mousemove") {
                    if (_interactions_drag_start_xy === undefined) {
                        return;
                    }
                }
            }

            // save the currently displayed stuff (before drawing new garbage)
            var c = get_canvas("Interaction");
            if (_interactions_backup_pixels === undefined) {
                _interactions_backup_pixels = c.getImageData(0, 0, _interactions_scale * _w,
                    _interactions_scale * _h);
            }

            // reset the visualization to the initial state (before drawing new garbage)
            c.putImageData(_interactions_backup_pixels, 0, 0);

            // get x,y
            var rect = c.canvas.getBoundingClientRect();
            var x = Math.floor((event.clientX - rect.left) / _interactions_scale);
            var y = Math.floor((event.clientY - rect.top) / _interactions_scale);

            if (event.type === "mousedown") {

                // single mouse press: save coordinates, since a new drag operation could start
                _interactions_drag_start_xy = [x, y];

            //} else if (event.type === "mousemove" || event.type === "mouseup") {
            } else if (event.type === "mouseup") {

                if (_interactions_drag_start_xy !== undefined) {
                    /*if (_interactions_drag_start_xy[0] !== x || _interactions_drag_start_xy[1] !== y) {

                        // a drag operation is finishing, fix negative stuff, supervision on the box, draw box!
                        var w = x - _interactions_drag_start_xy[0];
                        var h = y - _interactions_drag_start_xy[1];
                        if (w < 0) {
                            w = -w;
                        } else {
                            x = _interactions_drag_start_xy[0];
                        }
                        if (h < 0) {
                            h = -h;
                        } else {
                            y = _interactions_drag_start_xy[1];
                        }
                        draw_rect(c, x, y, w, h, '#00ff00', 5 / _interactions_scale);
                        _sup_x = x;
                        _sup_y = y;
                        _sup_w = w + 1;
                        _sup_h = h + 1;

                    } else {
                        if (event.type === "mouseup") {*/
                            // single mouse click: supervision on a single pixel (w=h=0), draw cross!
                            draw_cross(c, x, y, 20 / _interactions_scale, '#00ff00', 5 / _interactions_scale);
                            _sup_x = x;
                            _sup_y = y;
                            _sup_w = 1;
                            _sup_h = 1;
                        /*}
                    }*/
                }
            }

            // every time that the mouse button goes up, we kill every drag-start markers
            if (event.type === "mouseup") {
                _interactions_drag_start_xy = undefined;
            }
        }
    };

    // do not add handlers multiple times
    ctx.canvas.addEventListener('dblclick', mouse_event_handler, false);
    ctx.canvas.addEventListener('mousedown', mouse_event_handler, false);
    ctx.canvas.addEventListener('mouseup', mouse_event_handler, false);
    ctx.canvas.addEventListener('mousemove', mouse_event_handler, false);
}

// =============================================================================================================
// ================================== OPERATION-RELATED FUNCTIONS ==============================================
// =============================================================================================================

// this function is the one responsible of "getting-and-drawing-all-that-is-needed"
// in function of the action requested (play, stop, next, prev, refresh)
function do_action(action, it_is_an_inner_call) {
    if (it_is_an_inner_call !== undefined && it_is_an_inner_call) {
        log("do_action (inner call): " + action);
    } else {
        log("do_action: " + action);
    }

    var frame;

    if (action === 'refresh') {

        // in case of visualization changes, recreate all the canvas
        if (_views_visualization_changed === true) {
            _views_visualization_changed = false;
            reset_all_canvas(); // this will set _ajax_pending_requests to 0
            if (_sync_follow && it_is_an_inner_call) {
                _ajax_pending_requests = 1;
            }
        }

        // let's clear all the pending stuff
        clear_ajax_calls();

        // getting visualization options
        var params = get_GUI_options(); // also setting "_interactions_draw_style"
        var range_array;

        if (params.showVideo || (params.showMotion && !params.motionMap) || params.showInteractions
            || params.showSupervisions) {
            get_and_draw_frame(params.frame,
                !params.showVideo,
                params.showMotion && !params.motionMap,
                params.showFocus && params.showVideo,
                params.showInteractions, params.showSupervisions);
        }

        if (params.showInteractions || params.showSupervisions) {
            if (params.showInteractions)
                get_canvas("Interaction", _w, _h); // this is only to get a top position
            if (params.showSupervisions) {
                get_canvas("Supervisions", _w, _h); // this is only to get a top position
                get_and_draw_supervisions(params.frame);
            }
            get_and_draw_label_map();
        } else {
            hide_label_table(true);
        }

        if (params.showMotion) {
            get_and_draw_motion(params.frame, params.motionDrawStyle);
        }

        if (params.showPredictedMotion) {
            get_and_draw_additional_motion(params.frame, params.motionDrawStyle, "predicted_motion", "PredictedMotion");
        }

        if (params.showPredictedBackwardMotion) {
            get_and_draw_additional_motion(params.frame, params.motionDrawStyle, "predicted_backward_motion", "PredictedBackMotion");
        }

        /*get_and_draw_additional_motion(params.frame, params.motionDrawStyle, "fake_motion", "FakeMotion");*/

        if (params.showIHSMotion) {
            get_and_draw_additional_motion(params.frame, params.motionDrawStyle, "ihs_motion", "IHSMotion");
        }

        if (params.showPredictions || (params.showInteractions && _interactions_draw_style !== "none")) {
            range_array = $('#text_predictions_range').val().trim().split("-");
            range_array[0] = parseInt(range_array[0]);
            range_array[1] = parseInt(range_array[1]);

            get_and_draw_activations(params.frame, range_array, params.normalizePredictions, "sup-probs",
                !params.showPredictions, params.showInteractions && _interactions_draw_style !== "none");

            rename_predictor_related_canvas();
        }

        if (params.showDetails) {
            get_others(params.frame, params.layer);
        }

        for (var output_name in params.showGenericOutputs) {
            if (params.showGenericOutputs[output_name] === "IMAGE") {
                get_and_draw_image(params.frame, output_name);
            }
            if (params.showGenericOutputs[output_name] === "BINARY") {
                range_array = _text_range_generic_outputs[output_name].split("-");
                range_array[0] = parseInt(range_array[0]);
                if (range_array[1] !== 'last') {
                    range_array[1] = parseInt(range_array[1]);
                } else {
                    range_array[1] = -1;
                }

                get_and_draw_activations(params.frame, range_array,
                    params.normalizeGenericOutputs[output_name], output_name, false, false);
            }
        }
    } else if (action === 'prev') {
        do_action("stop", true);
        frame = get_currently_displayed_frame_number() - 1;
        set_currently_displayed_frame_number(Math.max(frame, 1));
        do_action("refresh", true);

    } else if (action === 'next') {
        if (it_is_an_inner_call === undefined) { // next is (inner) called to implement play, no need to stop
            do_action("stop", true);
        }
        if (!_sync_follow) {
            frame = get_currently_displayed_frame_number() + 1;
            set_currently_displayed_frame_number(frame);
            if (frame > get_currently_displayed_max_frame_number()) {
                set_currently_displayed_max_frame_number(frame);
            }
        }
        do_action("refresh", true);

    } else if (action === 'play') {
        //disable_GUI_options_about_what_to_show(true);
        do_action("stop", true);

        // activating the "play-thread" (timer)
        _play_thread = setInterval(function () {

                // avoid drawing again if drawing is in progress
                if (_ajax_pending_requests === 0 && _interactions_finished_drawing) {
                    if (_sync_follow) {
                        vprocessor_allow_processing_next_frame_only();
                    } else {
                        do_action("next", true); // inner call
                    }
                }
            },
            (1000.0 / get_currently_displayed_fps()));

    } else if (action === 'stop') {
        if (_play_thread !== undefined) {
            clearTimeout(_play_thread);
            log("Stopped play-thread!");
            _play_thread = undefined;
            //disable_GUI_options_about_what_to_show(false);
        }
    }
}

// activate sync mode (following)
function follow() {
    if (!_sync_follow) {
        _sync_follow = true;

        $('#button_sync_on').css('background-color', '#f47121');

        if (!_sync_paused) {
            do_action('play');
        } else {
            do_action('refresh');

            make_label_table_interactive(true);
        }
    }
}

// de-activate sync mode (not-following-anymore)
function stop_following() {
    if (_sync_follow) {
        _sync_follow = false;

        make_label_table_interactive(false);

        // TODO temporarily disabled
        /*
        warn_user_that_system_is_asking_for_a_supervision_and_adapt_GUI(false);
        */

        $('#button_sync_on').css('background-color', 'transparent');

        do_action('stop');
        if (!_sync_paused) {
            var repetitions = 0;
            var waitForDrawingToCompleteThenFreeTheVProcessor = setInterval(function () {
                    if (_ajax_pending_requests === 0 || repetitions >= 100) {
                        clearInterval(waitForDrawingToCompleteThenFreeTheVProcessor);
                        vprocessor_allow_processing();
                        reset_drawing_related_routines();
                    }
                },
                100);
        }
    }
}

// activate (sync) pause mode
function sync_pause_on() {
    if (!_sync_paused) {
        if (_sync_follow) {
            do_action('stop');
            var repetitions = 0;
            var waitForDrawingToComplete = setInterval(function () {
                    if (_ajax_pending_requests === 0 || repetitions >= 100) {
                        clearInterval(waitForDrawingToComplete);
                        reset_drawing_related_routines();

                        make_label_table_interactive(true);

                        // drawing FOA cross in the interaction window
                        if (is_canvas_existing("Interaction") && _foa !== undefined) {
                            let c = get_canvas("Interaction");
							_interactions_backup_pixels = c.getImageData(0, 0, _interactions_scale * _w, _interactions_scale * _h);
                            draw_cross(c, _foa.foay, _foa.foax, 20, '#00ff00', 5);
                        }
                    }
                },
                100);
        } else {
            vprocessor_disable_processing_asap();
        }

        _sync_paused = true;
        $('#button_sync_pause').css('background-color', '#f47121');
    }
}

// de-activate (sync) pause mode
function sync_pause_off() {
    if (_sync_paused) {
        if (_sync_follow) {
            _sup_x = undefined;
            _sup_y = undefined;
            _sup_w = undefined;
            _sup_h = undefined;

            make_label_table_interactive(false);

            if (_ajax_pending_requests === 0) {
                do_action('play');
            } else {

                // if there were pending requests, wait a moment and check again (evenutally forcing a reset)
                var waitForDrawingToCompleteThenPlay = setInterval(function () {
                        if (_ajax_pending_requests === 0) {
                            clearInterval(waitForDrawingToCompleteThenPlay);
                            do_action('play');
                        } else {
                            clearInterval(waitForDrawingToCompleteThenPlay);
                            reset_drawing_related_routines();
                            do_action('play');
                        }
                    },
                    100);
            }
        } else {

            // TODO temporarily disabled
            /*
            warn_user_that_system_is_asking_for_a_supervision_and_adapt_GUI(false);
            */

            vprocessor_allow_processing();
        }

        _sync_paused = false;
        $('#button_sync_pause').css('background-color', 'transparent');
    }
}

function template_predictions() {
    log("Switching to PREDICTIONS template");
    disable_unsupported_GUI_options();
    if (!$('#checkbox_video').is(":checked")) {
        $('#checkbox_video').click();
    }
    if (!$('#checkbox_motion').is(":checked")) {
        $('#checkbox_motion').click();
    }
    if (!$('#checkbox_details').is(":checked")) {
        $('#checkbox_details').click();
    }
    if (!$('#checkbox_focus').is(":checked")) {
        $('#checkbox_focus').click();
    }
    if (!$('#checkbox_predictions').is(":checked")) {
        $('#checkbox_predictions').click();
    }
    if (!$('#checkbox_interaction').is(":checked")) {
        $('#checkbox_interaction').click();
    }
    if ($('#checkbox_supervisions').is(":checked")) {
        $('#checkbox_supervisions').click();
    }
}

function template_all() {
    log("Switching to ALL template");
    disable_unsupported_GUI_options();
    if (!$('#checkbox_video').is(":checked")) {
        $('#checkbox_video').click();
    }
    if (!$('#checkbox_motion').is(":checked")) {
        $('#checkbox_motion').click();
    }
    if (!$('#checkbox_details').is(":checked")) {
        $('#checkbox_details').click();
    }
    if (!$('#checkbox_focus').is(":checked")) {
        $('#checkbox_focus').click();
    }
    if (!$('#checkbox_predictions').is(":checked")) {
        $('#checkbox_predictions').click();
    }
    if (!$('#checkbox_interaction').is(":checked")) {
        $('#checkbox_interaction').click();
    }
    if (!$('#checkbox_supervisions').is(":checked")) {
        $('#checkbox_supervisions').click();
    }
}

function template_none() {
    log("Switching to NONE template");
    disable_unsupported_GUI_options();
    if ($('#checkbox_video').is(":checked")) {
        $('#checkbox_video').click();
    }
    if ($('#checkbox_motion').is(":checked")) {
        $('#checkbox_motion').click();
    }
    if ($('#checkbox_details').is(":checked")) {
        $('#checkbox_details').click();
    }
    if ($('#checkbox_focus').is(":checked")) {
        $('#checkbox_focus').click();
    }
    if ($('#checkbox_predictions').is(":checked")) {
        $('#checkbox_predictions').click();
    }

    if ($('#checkbox_interaction').is(":checked")) {
        $('#checkbox_interaction').click();
    }
    if ($('#checkbox_supervisions').is(":checked")) {
        $('#checkbox_supervisions').click();
    }
}

function template_interaction() {
    log("Switching to INTERACTION template");
    if ($('#checkbox_video').is(":checked")) {
        $('#checkbox_video').click();
    }
    if ($('#checkbox_motion').is(":checked")) {
        $('#checkbox_motion').click();
    }
    if ($('#checkbox_details').is(":checked")) {
        $('#checkbox_details').click();
    }
    if ($('#checkbox_predictions').is(":checked")) {
        $('#checkbox_predictions').click();
    }
    if (!$('#checkbox_interaction').is(":checked")) {
        $('#checkbox_interaction').click();
    }
    if ($('#checkbox_focus').is(":checked")) {
        $('#checkbox_focus').click();
    }
    if ($('#checkbox_supervisions').is(":checked")) {
        $('#checkbox_supervisions').click();
    }
}

// check if a canvas exists in the view-table
function is_canvas_existing(caption) {
    var ctx = _views_caption_2_canvas_context[caption];
    return ctx !== undefined;
}

// TODO temporarily disabled
/*
function disable_GUI_options_about_interaction_view(disable_or_not) {
    if (disable_or_not) {
        $('#radio_predictions_max').attr('disabled', true);
        $('#radio_predictions_top').attr('disabled', true);
        $('#radio_predictions_none').attr('disabled', true);
        $('#radio_predictions_focus').attr('disabled', true);
    } else {
        $('#radio_predictions_max').removeAttr('disabled');
        $('#radio_predictions_top').removeAttr('disabled');
        $('#radio_predictions_none').removeAttr('disabled');
        $('#radio_predictions_focus').removeAttr('disabled');
    }
}
*/

// add a new row to the label table, composed of a pair "label, predictor ID"
function add_row_to_label_table(label, predictor, count) {
    p = parseInt(predictor);
    var c;
    if (p < 0)
        c = [0,0,0];
    else
        c = _colors[p % _colors.length];
    $('#label_table').append('<tr class="clickable" onclick="' +
        '{ ' +
        'let full_row_text = $(this).html(); ' + // "<td>label to be considered</td><td>predictor id</td>"
        'let label = full_row_text.substring(full_row_text.indexOf(\'>\')+1, ' +
        'full_row_text.indexOf(\'</\')); ' +
        'send_supervision(label.trim());' +
        '}' +
        '"><td>' + label + '</td><td>' + predictor + '</td><td id="_count_' + label + '">' + count +
        '</td><td style="background-color: rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></td></tr>');
}

// remove the rows from the label table, completely
function clear_label_table() {
    $('#label_table').html("");
}

// hide the label table
function hide_label_table(yes) {
    if (!yes) {
        if ($('#labelTableContainer').css("visibility") === "hidden") {
            $('#labelTableContainer').css("visibility", "visible")
        }
    } else {
        if ($('#labelTableContainer').css("visibility") === "visible") {
            $('#labelTableContainer').css("visibility", "hidden")
        }
    }
}

// make the rows of the label table "clickable" (or not)
function make_label_table_interactive(yes) {
    $('#label_table').find('tr').each(function () {
        if (yes) {
            $(this).attr('class', 'clickable');
            $(this).css("pointer-events", "auto");
            $('#new_class_button').css("pointer-events", "auto");
            $('#new_class_label').attr('disabled', false);
        } else {
            $(this).attr('class', 'not-clickable');
            $(this).css("pointer-events", "none");
            $('#new_class_button').css("pointer-events", "none");
            $('#new_class_label').attr('disabled', true);
        }
    });
}

// TODO temporarily disabled
/*
// implement those visual changes related to warning the user that the system is waiting for a supervision
function warn_user_that_system_is_asking_for_a_supervision_and_adapt_GUI(yes) {
    if (!yes) {
        // re-enabling options
        disable_GUI_options_about_interaction_view(false);

        // toggling highlight
        $('#controls').css("background-color", "transparent");
        $('#controls_legend').css("background-color", "transparent");
    } else {
        // force some visualization options
        $('#checkbox_interaction').prop('checked', true); // forcing interaction window
        $('#radio_predictions_focus').prop('checked', true); // forcing focus area
        disable_GUI_options_about_interaction_view(true); // disabling options

        // highlight
        $('#controls').css("background-color","yellow");
        $('#controls_legend').css("background-color","yellow");
    }
}
*/

// sending supervision
function send_supervision(label) {
    if (label.trim().length === 0) {
        alert("Missing class label!");
        return;
    }

    // removing the trailing *, that indicates a temporary entry in the label table
    if (label.endsWith("*")) {
        label = label.substr(0, label.length-1);
    }

    block_all_GUI(true);

    if (_sup_x === undefined || _sup_y === undefined || _sup_w === undefined || _sup_h === undefined) {
        //block_all_GUI(false);
        //alert("Which part of the frame you want to supervise?");
        //return;
        _sup_x = _foa.foay; // if no coordinates are provided, we supervise the FOA
        _sup_y = _foa.foax;
        _sup_w = 1;
        _sup_h = 1;
    }

    let sup = {
        x: _sup_x,
        y: _sup_y,
        w: _sup_w,
        h: _sup_h,
        class: label
    };

    send_command("supervise", JSON.stringify(sup),
        function () {
            after_having_sent_supervision(label);
            block_all_GUI(false);
        });
}

// what happens after a supervision has been sent to the system
function after_having_sent_supervision(label) {

    // adding a temporary row to the label table
    if (!(label in _sup_label_map) && !((label + "*") in _sup_label_map)) {
        label = label + "*";
        _sup_label_map[label] = -1;
        _sup_label_count[label] = -1;
        add_row_to_label_table(label, -1, -1);
    }

    if ($('#checkbox_sup_and_go').is(':checked')) {
        sync_pause_off();
    }
    // warn_user_that_system_is_asking_for_a_supervision_and_adapt_GUI(false); // TODO temporarily disabled
}

function selected_an_output_from_list() {
    _selected_output = $('#output_list option:selected').text();

    if ($('#output_list').val() === 'BINARY') {
        $('#text_output_range').prop('disabled', false);
        $('#checkbox_output_normalization').prop('disabled', false);
    }

    if ($('#output_list').val() === 'IMAGE') {
        $('#text_output_range').prop('disabled', true);
        $('#checkbox_output_normalization').prop('disabled', true);
    }

    // loading
    $('#checkbox_output_normalization').prop('checked', _normalize_generic_outputs[_selected_output]);
    $('#text_output_range').val(_text_range_generic_outputs[_selected_output]);
}

// TODO temporarily disabled
/*
// the system is asking for a supervision: prepare the GUI!
function system_initiative() {
    log("System initiative");

    // pause
    sync_pause_on();

    // highlight something in the GUI
    warn_user_that_system_is_asking_for_a_supervision_and_adapt_GUI(true);

    // setup a focus-area (segment) supervision (negative w or h)
    if (_interactions_foa !== undefined) {
        _sup_x = _interactions_foa.foa[1];
        _sup_y = _interactions_foa.foa[0];
    } else {
        _sup_x = undefined;
        _sup_y = undefined;
    }
    _sup_w = -1;
    _sup_h = -1;

    // update drawings
    do_action("refresh");
}
*/

// =============================================================================================================
// ======================================== ENTRY POINT (MAIN) =================================================
// =============================================================================================================

// this is what happen when the page loads for the first time (load options and set text field related things)
window.onload = function () {

    // get options and sync status (is the system paused? etc.)
    get_options(true);
    get_last_frame_number();
    get_and_react_to_sync_status(true);

    // disable the enter-key on the HTML page
    $('html').bind('keypress', function (e) {
        if (e.keyCode === 13) {
            return false;
        }
    });

    // ensure that the (visualized) FPS number is valid
    $('#text_fps').bind('keyup', function () {
        if ($('#text_fps').val().trim().length > 0) {
            var value = Number($('#text_fps').val());
            if (isNaN(value)) {
                $('#text_fps').val("1.0");
            } else {
                if (value <= 0) {
                    $('#text_fps').val("1.0");
                }
            }
        }
    });

    // ensure that the output range is valid and well ordered
    $('#text_output_range').bind('focusout', function () {
        if ($('#text_output_range').val().trim().length > 0) {
            var range_array = $('#text_output_range').val().trim().split("-");

            if (range_array.length === 2) {
                var value1 = Number(range_array[0]);
                var value2 = Number(range_array[1]);
                if (isNaN(value1)) {
                    value1 = 0;
                } else {
                    if (value1 < 0) {
                        value1 = 0;
                    }
                }
                if (isNaN(value2)) {
                    value2 = 'last';
                } else {
                    if (value2 < 0) {
                        value2 = 0;
                    }
                }
                if (value2 != 'last' && value1 > value2) {
                    var tmp = value1;
                    value1 = value2;
                    value2 = tmp;
                }
                $('#text_output_range').val(value1 + "-" + value2);
            } else {
                $('#text_output_range').val("0-last");
            }

            _text_range_generic_outputs[_selected_output] = $('#text_output_range').val();
        }
    });

    // hitting some keys toggles view changes
    $(document).keypress(function (e) {
        var id = $(e.target).attr("id");

        if (id !== "new_class_label") {
            if (e.which === 43) { // + = 43
                _interactions_scale = _interactions_scale + 1;
                if (_interactions_scale >= 4) {
                    _interactions_scale = 4;
                }
            }
            if (e.which === 45) { // - = 45
                _interactions_scale = _interactions_scale - 1;
                if (_interactions_scale === 0) {
                    _interactions_scale = 1;
                }
            }
            if (e.which === 97) { // a = 97
                template_all();
            }
            if (e.which === 110) { // n = 110
                template_none();
            }
            if (e.which === 112) { // p = 112
                template_predictions();
            }
            if (e.which === 105) { // i = 105
                template_interaction();
            }
        }
    });

    // catch the enter-key on the textbox of the label table to send a new label
    var ncl = document.getElementById("new_class_label");
    ncl.addEventListener("keydown", function (e) {
        if (e.keyCode === 13) {
            $('#new_class_button').click();
        }
    });

    // ensure that the predictions range is valid and well ordered
    $('#text_predictions_range').bind('focusout', function () {
        if ($('#text_predictions_range').val().trim().length > 0) {
            var range_array = $('#text_predictions_range').val().trim().split("-");

            if (range_array.length === 2) {
                var value1 = Number(range_array[0]);
                var value2 = Number(range_array[1]);
                if (isNaN(value1)) {
                    value1 = 0;
                } else {
                    if (value1 < 0) {
                        value1 = 0;
                    }
                    if (value1 >= _predictors) {
                        value1 = _predictors - 1;
                    }
                }
                if (isNaN(value2)) {
                    value2 = 0;
                } else {
                    if (value2 < 0) {
                        value2 = 0;
                    }
                    if (value2 >= _predictors) {
                        value2 = _predictors - 1;
                    }
                }
                if (value1 > value2) {
                    var tmp = value1;
                    value1 = value2;
                    value2 = tmp;
                }
                $('#text_predictions_range').val(value1 + "-" + value2);
            } else {
                $('#text_predictions_range').val("0-0");
            }

            if ($('#text_predictions_range').val() !== _predictions_range_text) {
                _views_visualization_changed = true;
            }
            _predictions_range_text = $('#text_predictions_range').val();
        }
    });
};