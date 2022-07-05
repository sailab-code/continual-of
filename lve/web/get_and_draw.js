// AJAX: getting the so called "others" data, that is a big JSON with several numerical details
function get_others(frame, layer) {
    log("Requested: all_stats_json");
    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/all_stats_json",
        data: {
            frame: frame,
            layer: layer,
            sync: _sync_follow === true ? 1 : 0
        },
        success: function (byte_str) {
            log("Returned: all_stats_json");

            try {
                var all_stats_json = JSON.parse(byte_str);
                document.getElementById("json").innerHTML = JSON.stringify(all_stats_json,
                    function(k,v){if(v instanceof Array)
                        return JSON.stringify(v,
                            function(k, v){if (typeof v === 'number')
                            {return parseFloat(v.toFixed(2));}return v;});
                    return v;},
                    2);
                _ajax_pending_requests--;
            } catch (err) {
                document.getElementById("json").innerHTML = "Not found!";
                _ajax_pending_requests--; // this guy always returns something (even "{}")
                
            }
        },
        error: function () {
            log("Returned: all_stats_json (error)");
            document.getElementById("json").innerHTML = "Not found!";
            _ajax_pending_requests--; // this guy always returns something (even "{}")
            _ajax_requests_that_did_not_failed--;
        }
    }));
}

// AJAX: get and draw the video frame
function get_and_draw_frame(frame, get_only, motion_lines_shown, show_focus, interactions_shown, supervisions_shown) {
    log("Requested: frames");
    _frame_pixels = undefined;
    var ctx;

    if (!get_only) {
        ctx = get_canvas("Video", _w, _h); // ensure this is immediately returned (to gain a top row position)
    } else {
        ctx = document.createElement('canvas').getContext('2d'); // hidden canvas
        ctx.canvas.width = _w;
        ctx.canvas.height = _h;
    }

    // getting focus of attention
    if (show_focus) {
        get_and_draw_focus(frame);
    }

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    // we can request the frame in two ways: if it is loaded from disk, then it is a PNG image
    // if it is asked in sync mode (following), then it is a numpy array
    if (!_sync_follow) {
        _pending_ajax_calls.push($.ajax({
            type: "GET",
            url: "/frames",
            dataType: "arraybuffer",
            data: {
                frame: frame,
                sync: 0
            },
            success: function (array_buffer) {
                log("Returned: frames");
                var blb;
                var img;
                try {
                    blb = new Blob([array_buffer], {type: 'image/png'});
                    img = new Image();
                } catch (err) {
                    handle_ajax_draw_error(ctx, get_only, true);
                    return;
                }

                img.onload = function () {
                    ctx.drawImage(img, 0, 0, img.width, img.height);
                    var imgData = ctx.getImageData(0, 0, img.width, img.height);
                    _frame_pixels = imgData.data;

                    if (!get_only) {
                        ctx.putImageData(imgData, 0, 0);

                        // eventually completing dependent drawing operations: focus
                        if (show_focus && _foa !== undefined) {
                            _draw_focus();
                        }
                    }

                    // eventually completing dependent drawing operations: motion
                    if (motion_lines_shown) {
                        if (_motion !== undefined) {
                            _draw_motion_lines();
                        }
                    }

                    // eventually completing dependent drawing operations: interactions
                    if (interactions_shown) {
                        if (_interactions_draw_style === "none" || _predictions !== undefined) {
                            _draw_interactions();
                        }
                    }

                    // eventually completing dependent drawing operations: supervisions
                    if (supervisions_shown) {
                        if (_sup_indices !== undefined && _sup_targets !== undefined) {
                            _draw_supervisions()
                        }
                    }

                    _ajax_pending_requests--;
                };

                img.src = (window.URL || window.webkitURL).createObjectURL(blb);
            },
            error: function () {
                log("Returned: frames (error)");
                _frame_pixels = -1;
                handle_ajax_draw_error(ctx, get_only, true);
            }
        }));
    } else {
        _pending_ajax_calls.push($.ajax({
            type: "GET",
            url: "/frames",
            dataType: "arraybuffer",
            data: {
                frame: frame,
                sync: 1
            },
            success: function (array_buffer) {
                log("Returned: frames");
                var numPyArray;
                try {
                    numPyArray = getDataFromNumPyArray(array_buffer);
                } catch (err) {
                    handle_ajax_draw_error(ctx, get_only, true);
                    return;
                }
                var w = numPyArray.w;
                var h = numPyArray.h;
                var c = numPyArray.c;
                var z = 1;
                if (c < 3) z = 0;
                var zz = 2 * z;
                var array_of_floats = numPyArray.data;

                var imgData = ctx.createImageData(w, h);
                var data = imgData.data;
                var k = 0;
                for (var i = 0; i < array_of_floats.length; i = i + c) {
                    data[k] = array_of_floats[i + zz]; // R
                    data[k + 1] = array_of_floats[i + z]; // G
                    data[k + 2] = array_of_floats[i]; // B
                    data[k + 3] = 255; // A
                    k = k + 4;
                }

                _frame_pixels = imgData.data;

                if (!get_only) {
                    ctx.putImageData(imgData, 0, 0);

                    // eventually completing dependent drawing operations: focus
                    if (show_focus && _foa !== undefined) {
                        _draw_focus();
                    }
                }

                // eventually completing dependent drawing operations: motion
                if (motion_lines_shown) {
                    if (_motion !== undefined) {
                        _draw_motion_lines();
                    }
                }

                // eventually completing dependent drawing operations: interactions
                if (interactions_shown) {
                    if (_interactions_draw_style === "none" || _predictions !== undefined) {
                        _draw_interactions();
                    }
                }

                // eventually completing dependent drawing operations: supervisions
                if (supervisions_shown) {
                    if (_sup_indices !== undefined && _sup_targets !== undefined) {
                        _draw_supervisions()
                    }
                }

                _ajax_pending_requests--;
            },
            error: function () {
                log("Returned: frames (error)");
                _frame_pixels = -1;
                handle_ajax_draw_error(ctx, get_only, true);
            }
        }));
    }
}

// AJAX: get and draw some image data (NOT the input video frame)
function get_and_draw_image(frame, request) {
    log("Requested: " + request);
    var ctx;

    var caption = request[0].toUpperCase() + request.slice(1);
    ctx = get_canvas(caption, _w, _h); // ensure this is immediately returned (to gain a top row position)

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    // we can request the image in two ways: if it is loaded from disk, then it is a PNG image
    // if it is asked in sync mode (following), then it is a numpy array
    if (!_sync_follow) {
        _pending_ajax_calls.push($.ajax({
            type: "GET",
            url: "/" + request,
            dataType: "arraybuffer",
            data: {
                frame: frame,
                sync: 0
            },
            success: function (array_buffer) {
                log("Returned: " + request);
                var blb;
                var img;
                try {
                    blb = new Blob([array_buffer], {type: 'image/png'});
                    img = new Image();
                } catch (err) {
                    handle_ajax_draw_error(ctx, false, true);
                    return;
                }

                img.onload = function () {
                    ctx.drawImage(img, 0, 0, img.width, img.height);
                    var imgData = ctx.getImageData(0, 0, img.width, img.height);
                    ctx.putImageData(imgData, 0, 0);

                    _ajax_pending_requests--;
                };

                img.src = (window.URL || window.webkitURL).createObjectURL(blb);
            },
            error: function () {
                log("Returned: " + request + " (error)");
                handle_ajax_draw_error(ctx, false, true);
            }
        }));
    } else {
        _pending_ajax_calls.push($.ajax({
            type: "GET",
            url: "/" + request,
            dataType: "arraybuffer",
            data: {
                frame: frame,
                sync: 1
            },
            success: function (array_buffer) {
                log("Returned: " + request);
                var numPyArray;
                try {
                    numPyArray = getDataFromNumPyArray(array_buffer);
                } catch (err) {
                    handle_ajax_draw_error(ctx, false, true);
                    return;
                }
                var w = numPyArray.w;
                var h = numPyArray.h;
                var c = numPyArray.c;
                var z = 1;
                if (c < 3) z = 0;
                var zz = 2 * z;
                var array_of_floats = numPyArray.data;

                var imgData = ctx.createImageData(w, h);
                var data = imgData.data;
                var k = 0;
                for (var i = 0; i < array_of_floats.length; i = i + c) {
                    data[k] = array_of_floats[i + zz]; // R
                    data[k + 1] = array_of_floats[i + z]; // G
                    data[k + 2] = array_of_floats[i]; // B
                    data[k + 3] = 255; // A
                    k = k + 4;
                }

                ctx.putImageData(imgData, 0, 0);

                _ajax_pending_requests--;
            },
            error: function () {
                log("Returned: " + request + " (error)");
                handle_ajax_draw_error(ctx, false, true);
            }
        }));
    }
}

// AJAX: get and draw the motion field (draw style can be "map" or "lines")
function get_and_draw_motion(frame, draw_style) {
    log("Requested: motion");

    _motion = undefined;

    var ctx = get_canvas("Motion", _w, _h); // ensure this is immediately returned (to gain a top row position)

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/motion",
        dataType: "arraybuffer",
        data: {
            frame: frame,
            sync: _sync_follow === true ? 1 : 0
        },
        success: function (array_buffer) {
            log("Returned: motion");
            var numPyArray;
            try {
                numPyArray = getDataFromNumPyArray(array_buffer);
            } catch (err) {
                handle_ajax_draw_error(ctx, false, true);
                return;
            }
            var ww = numPyArray.w;
            var hh = numPyArray.h;
            var c_first = numPyArray.c_first;
            var array_of_floats = numPyArray.data;

            if (draw_style === 'map') {
                var imgData = ctx.createImageData(ww, hh);
                var data = imgData.data;

                var vx;
                var vy;
                var k = 0;

                var h, s, v;
                var rgb;
                var mag;
                var ang;
                var min_mag = Number.POSITIVE_INFINITY;
                var max_mag = Number.NEGATIVE_INFINITY;
                var i = 0;

                if (!c_first) {
                    for (i = 0; i < array_of_floats.length; i = i + 2) {
                        vx = array_of_floats[i];
                        vy = array_of_floats[i + 1];

                        mag = Math.sqrt(vx * vx + vy * vy);
                        ang = Math.atan2(vy, vx) * (180 / Math.PI);
                        if (ang < 0.0) {
                            ang = ang + 360.0;
                        }
                        min_mag = Math.min(mag, min_mag);
                        max_mag = Math.max(mag, max_mag);

                        array_of_floats[i] = ang; // [0,360]
                        array_of_floats[i + 1] = mag; // [0,whatever]
                    }

                    max_mag = max_mag - min_mag;

                    for (i = 0; i < array_of_floats.length; i = i + 2) {
                        h = array_of_floats[i] / 360.0; // [0,1]
                        s = 1.0; // [0,1]
                        v = ((array_of_floats[i + 1] - min_mag) / max_mag); // [0,1]
                        v = Math.max(Math.min(v, 1.0), 0.0); // ensure [0,1]

                        rgb = hsvToRgb(h, s, v);

                        data[k] = rgb[0]; // R
                        data[k + 1] = rgb[1]; // G
                        data[k + 2] = rgb[2]; // B
                        data[k + 3] = 255; // A
                        k = k + 4;
                    }
                } else {
                    var n = array_of_floats.length / 2;

                    for (i = 0; i < n; i = i + 1) {
                        vx = array_of_floats[i];
                        vy = array_of_floats[i + n];

                        mag = Math.sqrt(vx * vx + vy * vy);
                        ang = Math.atan2(vy, vx) * (180 / Math.PI);
                        if (ang < 0.0) {
                            ang = ang + 360.0;
                        }
                        min_mag = Math.min(mag, min_mag);
                        max_mag = Math.max(mag, max_mag);

                        array_of_floats[i] = ang; // [0,360]
                        array_of_floats[i + n] = mag; // [0,whatever]
                    }

                    max_mag = max_mag - min_mag;

                    for (i = 0; i < n; i = i + 1) {
                        h = array_of_floats[i] / 360.0; // [0,1]
                        s = 1.0; // [0,1]
                        v = ((array_of_floats[i + n] - min_mag) / max_mag); // [0,1]
                        v = Math.max(Math.min(v, 1.0), 0.0); // ensure [0,1]

                        rgb = hsvToRgb(h, s, v);

                        data[k] = rgb[0]; // R
                        data[k + 1] = rgb[1]; // G
                        data[k + 2] = rgb[2]; // B
                        data[k + 3] = 255; // A
                        k = k + 4;
                    }
                }

                ctx.putImageData(imgData, 0, 0);

            } else if (draw_style === 'lines') {
                _motion = array_of_floats;
                _motion_c_first = c_first;

                if (_frame_pixels !== undefined) {
                    _draw_motion_lines();
                }
            }

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: motion (error)");
            handle_ajax_draw_error(ctx, false, true);
        }
    }));
}

function get_and_draw_additional_motion(frame, draw_style, name, canvas_name) {
    log("Requested: "+name);

    _motion = undefined;

    var ctx = get_canvas(canvas_name, _w, _h); // ensure this is immediately returned (to gain a top row position)

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/"+name,
        dataType: "arraybuffer",
        data: {
            frame: frame,
            sync: _sync_follow === true ? 1 : 0
        },
        success: function (array_buffer) {
            log("Returned: "+name);
            var numPyArray;
            try {
                numPyArray = getDataFromNumPyArray(array_buffer);
            } catch (err) {
                handle_ajax_draw_error(ctx, false, true);
                return;
            }
            var ww = numPyArray.w;
            var hh = numPyArray.h;
            var c_first = numPyArray.c_first;
            var array_of_floats = numPyArray.data;

            if (draw_style === 'map') {
                var imgData = ctx.createImageData(ww, hh);
                var data = imgData.data;

                var vx;
                var vy;
                var k = 0;

                var h, s, v;
                var rgb;
                var mag;
                var ang;
                var min_mag = Number.POSITIVE_INFINITY;
                var max_mag = Number.NEGATIVE_INFINITY;
                var i = 0;

                if (!c_first) {
                    for (i = 0; i < array_of_floats.length; i = i + 2) {
                        vx = array_of_floats[i];
                        vy = array_of_floats[i + 1];

                        mag = Math.sqrt(vx * vx + vy * vy);
                        ang = Math.atan2(vy, vx) * (180 / Math.PI);
                        if (ang < 0.0) {
                            ang = ang + 360.0;
                        }
                        min_mag = Math.min(mag, min_mag);
                        max_mag = Math.max(mag, max_mag);

                        array_of_floats[i] = ang; // [0,360]
                        array_of_floats[i + 1] = mag; // [0,whatever]
                    }

                    max_mag = max_mag - min_mag;

                    for (i = 0; i < array_of_floats.length; i = i + 2) {
                        h = array_of_floats[i] / 360.0; // [0,1]
                        s = 1.0; // [0,1]
                        v = ((array_of_floats[i + 1] - min_mag) / max_mag); // [0,1]
                        v = Math.max(Math.min(v, 1.0), 0.0); // ensure [0,1]

                        rgb = hsvToRgb(h, s, v);

                        data[k] = rgb[0]; // R
                        data[k + 1] = rgb[1]; // G
                        data[k + 2] = rgb[2]; // B
                        data[k + 3] = 255; // A
                        k = k + 4;
                    }
                } else {
                    var n = array_of_floats.length / 2;

                    for (i = 0; i < n; i = i + 1) {
                        vx = array_of_floats[i];
                        vy = array_of_floats[i + n];

                        mag = Math.sqrt(vx * vx + vy * vy);
                        ang = Math.atan2(vy, vx) * (180 / Math.PI);
                        if (ang < 0.0) {
                            ang = ang + 360.0;
                        }
                        min_mag = Math.min(mag, min_mag);
                        max_mag = Math.max(mag, max_mag);

                        array_of_floats[i] = ang; // [0,360]
                        array_of_floats[i + n] = mag; // [0,whatever]
                    }

                    max_mag = max_mag - min_mag;

                    for (i = 0; i < n; i = i + 1) {
                        h = array_of_floats[i] / 360.0; // [0,1]
                        s = 1.0; // [0,1]
                        v = ((array_of_floats[i + n] - min_mag) / max_mag); // [0,1]
                        v = Math.max(Math.min(v, 1.0), 0.0); // ensure [0,1]

                        rgb = hsvToRgb(h, s, v);

                        data[k] = rgb[0]; // R
                        data[k + 1] = rgb[1]; // G
                        data[k + 2] = rgb[2]; // B
                        data[k + 3] = 255; // A
                        k = k + 4;
                    }
                }

                ctx.putImageData(imgData, 0, 0);

            } else if (draw_style === 'lines') {
                _motion = array_of_floats;
                _motion_c_first = c_first;

                if (_frame_pixels !== undefined) {
                    _draw_motion_lines();
                }
            }

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: motion (error)");
            handle_ajax_draw_error(ctx, false, true);
        }
    }));
}

// function that is used to complete the motion-line-based drawing when
// both the motion field and the video frame have been received
function _draw_motion_lines() {
    var ctx = get_canvas("Motion");
    var w = _w;
    var h = _h;
    var imgData = ctx.createImageData(w, h);
    var data = imgData.data;

    copyAndGoGray(data, _frame_pixels);

    var vx;
    var vy;

    ctx.putImageData(imgData, 0, 0);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#00ff00';
    ctx.lineCap = 'butt';

    var step_size = 16;
    var w2 = 2 * w;
    var p2 = 2 * Math.PI;
    var offset;
    var y, x;

    if (!_motion_c_first) {
        w2 = 2 * w;

        for (y = 0; y < h; y = y + step_size) {
            offset = y * w2;
            for (x = 0; x < w; x = x + step_size) {
                vx = _motion[offset + 2 * x + 0];
                vy = _motion[offset + 2 * x + 1];

                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(Math.max(Math.min(Math.floor(x + vx), w), 0),
                    Math.max(Math.min(Math.floor(y + vy), h), 0));
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(x, y, 1, 0, p2);
                ctx.stroke();
            }
        }
    } else {
        var n = _motion.length / 2;

        for (y = 0; y < h; y = y + step_size) {
            offset = y * w;
            for (x = 0; x < w; x = x + step_size) {
                vx = _motion[offset + x];
                vy = _motion[offset + x + n];

                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(Math.max(Math.min(Math.floor(x + vx), w), 0),
                    Math.max(Math.min(Math.floor(y + vy), h), 0));
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(x, y, 1, 0, p2);
                ctx.stroke();
            }
        }
    }
}

// AJAX: get and draw the focus of attention (foa) on the video canvas
function get_and_draw_focus(frame) {
    log("Requested: stats.foa (in get_and_draw_focus)");
    _foa = undefined;

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/stats.worker",
        data: {
            frame: frame,
            sync: _sync_follow ? 1 : 0
        },
        success: function (byte_str) {
            log("Returned: stats.worker (in get_and_draw_focus)");
            try {
                _foa = JSON.parse(byte_str);
            } catch (err) {
                _foa = undefined;
            }

            if (_frame_pixels !== undefined) {
                _draw_focus();
            }

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: stats.worker (in get_and_draw_focus) (error)");
            _foa = -1;
            handle_ajax_draw_error(null, true, true);
        }
    }));
}

// function that is used to complete the focus of attention-related drawing when
// both the the video frame and the focus of attention have been received
function _draw_focus() {
    var ctx = get_canvas("Video");

    if (_frame_pixels === -1 || _foa === undefined) {
        return;
    }

    // do not add handlers multiple times
    if (!_foa_registered_mouse) {

        // handler of all mouse-related interactions with this view
        var mouse_event_handler = function (event) {

            // only catch events when in sync (following) mode
            if (_sync_follow) {

                // clicking the mouse button
                if (event.type === "mousedown") {
                    if (event.which !== 3) {
                        var rect = get_canvas("Video").canvas.getBoundingClientRect();
                        var x = Math.floor(event.clientX - rect.left);
                        var y = Math.floor(event.clientY - rect.top);
                        send_command("reset_foa", JSON.stringify({x: x, y: y}));
                    }
                }
            }
        };

        _foa_registered_mouse = true;
        ctx.canvas.addEventListener('mousedown', mouse_event_handler, false);
    }

    if (!_foa.saccade) {
        draw_cross(ctx, _foa.foay, _foa.foax, 20, '#ff0000', 5);
    } else {
        draw_cross(ctx, _foa.foay, _foa.foax, 20, '#0000ff', 5);
    }
}

// AJAX: getting system label map
function get_and_draw_label_map() {
    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    log("Requested: supervision_map_and_counts");
    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/supervision_map_and_counts",
        success: function (byte_str) {
            log("Returned: supervision_map_and_counts");

            var map_and_count = JSON.parse(byte_str);

            var received_map = map_and_count.supervision_map;
            var received_counts = map_and_count.supervision_count;
            var labels = Object.keys(received_map).sort();

            var update_full_table = false;

            // if just one label has changed, update the whole label table
            var aKeys = Object.keys(_sup_label_map).sort();
            if (JSON.stringify(aKeys) !== JSON.stringify(labels)) {
                update_full_table = true;
            }

            var i = 0;

            if (update_full_table) {

                // full update
                _sup_label_map = received_map;
                _sup_label_count = received_counts;
                _sup_label_map_inv = {};

                clear_label_table();
                for (i = 0; i < labels.length; i++) {
                    add_row_to_label_table(labels[i], received_map[labels[i]], received_counts[labels[i]]);
                    _sup_label_map_inv[received_map[labels[i]]] = labels[i]
                }
                make_label_table_interactive(false);
            } else {

                // updating counts only (if needed)
                for (i = 0; i < labels.length; i++) {
                    if (received_counts[labels[i]] !== _sup_label_count[labels[i]]) {
                        $('#_count_' + labels[i]).html(received_counts[labels[i]]);
                        _sup_label_count[labels[i]] = received_counts[labels[i]];
                    }
                }
            }

            hide_label_table(false);

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: supervision_map_and_counts (error)");
            handle_ajax_draw_error(null, true, true);
        }
    }));
}

// AJAX: get and drawn activation scores (different types)
function get_and_draw_activations(frame, range, normalize, request, get_only, interactions_shown) {
    let url;
    let dt;
    let caption;

    let data_in_01 = request.includes("prob");
    let requested_predictions = request == "sup-probs"

    log("Requested: " + request);
    url = "/" + request;
    dt = {
        frame: frame,
        sync: _sync_follow === true ? 1 : 0
    };

    if (!requested_predictions)
        caption = request[0].toUpperCase() + request.slice(1);
    else
        caption = "Predictor";

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: url,
        dataType: "arraybuffer",
        data: dt,
        success: function (array_buffer) {
            var from, to;
            log("Returned: " + request);

            var numPyArray;
            var f;
            var ctx;

            try {
                numPyArray = getDataFromNumPyArray(array_buffer);
            } catch (err) {
                from = range[0];
                f = from;
                var done_something = false;

                let _caption = caption + " " + f;

                while (is_canvas_existing(_caption)) {
                    ctx = get_canvas(_caption, _w, _h);
                    handle_ajax_draw_error(ctx, get_only, f === from);
                    f = f + 1;
                    _caption = caption + " " + f;
                    done_something = true;
                }

                if (!done_something) {
                    handle_ajax_draw_error(ctx, get_only, f === from);
                }
                return;
            }

            var w = numPyArray.w;
            var h = numPyArray.h;
            var c = numPyArray.c;
            var c_first = numPyArray.c_first;
            var whc = w * h * c;
            var array_of_floats = numPyArray.data;

            if (!get_only) {
                from = range[0];
                to = range[1] >= 0 ? range[1] : (c - 1);
                range[1] = to; // needed to handle errors below
                for (f = from; f <= to; f++) {
                    let _caption = caption + " " + f;

                    ctx = get_canvas(_caption, w, h);
                    var imgData = ctx.createImageData(w, h);
                    var data = imgData.data;

                    var val;
                    var k = 0;

                    var min_val = Number.POSITIVE_INFINITY;
                    var max_val = Number.NEGATIVE_INFINITY;

                    var i;

                    if (!c_first) {
                        if (normalize === false) {
                            if (data_in_01) {
                                for (i = f; i < whc; i = i + c) {
                                    val = 255.0 * array_of_floats[i];

                                    data[k] = val; // R
                                    data[k + 1] = val; // G
                                    data[k + 2] = val; // B
                                    data[k + 3] = 255; // A
                                    k = k + 4;
                                }
                            } else {
                                for (i = f; i < whc; i = i + c) {

                                    // these two lines will plot in [0,255] the values that are in [-0.5,0.5]
                                    val = array_of_floats[i] + 0.5
                                    val = 255.0 * Math.max(0.0, Math.min(1.0, val));

                                    data[k] = val; // R
                                    data[k + 1] = val; // G
                                    data[k + 2] = val; // B
                                    data[k + 3] = 255; // A
                                    k = k + 4;
                                }
                            }
                        } else {
                            for (i = f; i < whc; i = i + c) {
                                val = array_of_floats[i];

                                min_val = Math.min(val, min_val);
                                max_val = Math.max(val, max_val);
                            }

                            max_val = max_val - min_val;

                            for (i = f; i < whc; i = i + c) {
                                val = 255 * ((array_of_floats[i] - min_val) / (max_val));

                                data[k] = val; // R
                                data[k + 1] = val; // G
                                data[k + 2] = val; // B
                                data[k + 3] = 255; // A
                                k = k + 4;
                            }
                        }
                    } else {
                        var wh = w * h;
                        var f1wh = (f + 1) * wh;

                        if (normalize === false) {
                            if (data_in_01) {
                                for (i = f * wh; i < f1wh; i = i + 1) {
                                    val = 255.0 * array_of_floats[i];

                                    data[k] = val; // R
                                    data[k + 1] = val; // G
                                    data[k + 2] = val; // B
                                    data[k + 3] = 255; // A
                                    k = k + 4;
                                }
                            } else {
                                for (i = f * wh; i < f1wh; i = i + 1) {

                                    // these two lines will plot in [0,255] the data that are in [-0.5,0.5]
                                    val = array_of_floats[i] + 0.5;
                                    val = 255.0 * Math.max(0.0, Math.min(1.0, val));

                                    data[k] = val; // R
                                    data[k + 1] = val; // G
                                    data[k + 2] = val; // B
                                    data[k + 3] = 255; // A
                                    k = k + 4;
                                }
                            }
                        } else {
                            for (i = f * wh; i < f1wh; i = i + 1) {
                                val = array_of_floats[i];

                                min_val = Math.min(val, min_val);
                                max_val = Math.max(val, max_val);
                            }

                            max_val = max_val - min_val;

                            for (i = f * wh; i < f1wh; i = i + 1) {
                                val = 255 * ((array_of_floats[i] - min_val) / (max_val));

                                data[k] = val; // R
                                data[k + 1] = val; // G
                                data[k + 2] = val; // B
                                data[k + 3] = 255; // A
                                k = k + 4;
                            }
                        }
                    }

                    ctx.putImageData(imgData, 0, 0);
                }
            }

            // eventually completing dependent operations: interactions
            if (requested_predictions) {
                _predictions = array_of_floats;
                _predictions_c_first = c_first;

                if (interactions_shown) {
                    if (_frame_pixels !== undefined) {
                        _draw_interactions();
                    }
                }
            }

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: " + request + " (error)");
            if (requested_predictions) {
                _predictions = -1; // clearing
            }

            let from = range[0];
            let f = from;
            let ctx;
            let done_something = false;

            let _caption = caption + " " + f;

            while (!get_only && is_canvas_existing(_caption)) {
                ctx = get_canvas(_caption, _w, _h);
                handle_ajax_draw_error(ctx, get_only, f === from);
                f = f + 1;
                _caption = caption + " " + f;
                done_something = true;
            }

            if (!done_something) {
                handle_ajax_draw_error(ctx, true, f === from);
            }
        }
    }));
}

