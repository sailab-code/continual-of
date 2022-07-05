// function that is used to complete the interaction-related drawing when
// both the predictions, the video frame have been received
function _draw_interactions() {
    var ctx = get_canvas("Interaction", _w, _h);
    _interactions_backup_pixels = undefined;
    _interactions_finished_drawing = false;

    var imgData = ctx.createImageData(_w, _h);
    var data = imgData.data;

    if (_frame_pixels === -1) {
        drawNotFoundOrError(ctx, ctx.canvas.width, ctx.canvas.height);
        return;
    }

    copyAndGoGray(data, _frame_pixels);
    let pred_map = new Map();

    if (_predictions !== -1) {
        if (_interactions_draw_style === "none") {
            // nothing to do here!
        } else {
            const array_of_floats = _predictions;
            const whc = _w * _h * _predictors;
            const c = _predictors;
            let m = 0;
            let a = 0;
            const beta = 0.5;
            let color;
            const umbeta = 1.0 - beta;
            let k, i, jx;
            let freq, sorted_freq, cut;
            let gray;

            if (!_predictions_c_first) {
                if (_interactions_draw_style === "max") {
                    k = 0;
                    for (i = 0; i < whc; i = i + c) {
                        m = array_of_floats[i];
                        a = 0;
                        for (jx = 1; jx < c; jx++) {
                            if (m < array_of_floats[i + jx]) {
                                m = array_of_floats[i + jx];
                                a = jx;
                            }
                        }

                        if (pred_map.get(a) === undefined) {
                            pred_map.set(a, i);
                        }

                        if (m !== 0.) {
                            color = _colors[a % 10];
                            data[k] = beta * data[k] + (umbeta) * color[0]; // R
                            data[k + 1] = beta * data[k + 1] + (umbeta) * color[1]; // G
                            data[k + 2] = beta * data[k + 2] + (umbeta) * color[2]; // B
                            data[k + 3] = 255; // A
                            k = k + 4;
                        } else {
                            gray = (data[k] + data[k + 1] + data[k + 2]) / 3.
                            data[k] = gray; // R
                            data[k + 1] = gray; // G
                            data[k + 2] = gray; // B
                            data[k + 3] = 255; // A
                            k = k + 4;
                        }
                    }
                } else if (_interactions_draw_style === "top3") {
                    k = 0;
                    freq = [];
                    for (jx = 0; jx < c; jx++) {
                        freq[jx] = 0;
                    }
                    for (i = 0; i < whc; i = i + c) {
                        m = array_of_floats[i];
                        a = 0;
                        for (jx = 1; jx < c; jx++) {
                            if (m < array_of_floats[i + jx]) {
                                m = array_of_floats[i + jx];
                                a = jx;
                            }
                        }
                        freq[a] = freq[a] + 1;
                    }
                    sorted_freq = freq.slice(0).sort();
                    cut = sorted_freq[Math.max(c - 3, 0)];

                    for (i = 0; i < whc; i = i + c) {
                        m = array_of_floats[i];
                        a = 0;
                        for (jx = 1; jx < c; jx++) {
                            if (m < array_of_floats[i + jx]) {
                                m = array_of_floats[i + jx];
                                a = jx;
                            }
                        }

                        if (freq[a] < cut) {
                            if (pred_map.get(a) === undefined) {
                                pred_map.set(a, i);
                            }

                            data[k] = beta * data[k]; // R
                            data[k + 1] = beta * data[k + 1]; // G
                            data[k + 2] = beta * data[k + 2]; // B
                            data[k + 3] = 255; // A
                        } else {
                            color = _colors[a % 10];
                            data[k] = beta * data[k] + (umbeta) * color[0]; // R
                            data[k + 1] = beta * data[k + 1] + (umbeta) * color[1]; // G
                            data[k + 2] = beta * data[k + 2] + (umbeta) * color[2]; // B
                            data[k + 3] = 255; // A
                        }

                        k = k + 4;
                    }
                }
            } else {
                const wh = _w * _h;

                if (_interactions_draw_style === "max") {
                    k = 0;

                    for (i = 0; i < wh; i = i + 1) {
                        m = array_of_floats[i];
                        a = 0;
                        for (jx = wh; jx < whc; jx = jx + wh) {
                            if (m < array_of_floats[i + jx]) {
                                m = array_of_floats[i + jx];
                                a = jx / wh;
                            }
                        }

                        if (pred_map.get(a) === undefined) {
                            pred_map.set(a, i);
                        }

                        if (m !== 0.) {
                            color = _colors[a % 10];
                            data[k] = beta * data[k] + (umbeta) * color[0]; // R
                            data[k + 1] = beta * data[k + 1] + (umbeta) * color[1]; // G
                            data[k + 2] = beta * data[k + 2] + (umbeta) * color[2]; // B
                            data[k + 3] = 255; // A
                            k = k + 4;
                        } else {
                            gray = (data[k] + data[k + 1] + data[k + 2]) / 3.
                            data[k] = gray; // R
                            data[k + 1] = gray; // G
                            data[k + 2] = gray; // B
                            data[k + 3] = 255; // A
                            k = k + 4;
                        }
                    }
                } else if (_interactions_draw_style === "top3") {
                    k = 0;
                    freq = [];
                    for (jx = 0; jx < c; jx++) {
                        freq[jx] = 0;
                    }
                    for (i = 0; i < wh; i = i + 1) {
                        m = array_of_floats[i];
                        a = 0;
                        for (jx = wh; jx < whc; jx = jx + wh) {
                            if (m < array_of_floats[i + jx]) {
                                m = array_of_floats[i + jx];
                                a = jx / wh;
                            }
                        }
                        freq[a] = freq[a] + 1;
                    }
                    sorted_freq = freq.slice(0).sort();
                    cut = sorted_freq[Math.max(c - 3, 0)];

                    for (i = 0; i < wh; i = i + 1) {
                        m = array_of_floats[i];
                        a = 0;
                        for (jx = wh; jx < whc; jx = jx + wh) {
                            if (m < array_of_floats[i + jx]) {
                                m = array_of_floats[i + jx];
                                a = jx / wh;
                            }
                        }

                        if (freq[a] < cut) {
                            if (pred_map.get(a) === undefined) {
                                pred_map.set(a, i);
                            }

                            data[k] = beta * data[k]; // R
                            data[k + 1] = beta * data[k + 1]; // G
                            data[k + 2] = beta * data[k + 2]; // B
                            data[k + 3] = 255; // A
                        } else {
                            color = _colors[a % 10];
                            data[k] = beta * data[k] + (umbeta) * color[0]; // R
                            data[k + 1] = beta * data[k + 1] + (umbeta) * color[1]; // G
                            data[k + 2] = beta * data[k + 2] + (umbeta) * color[2]; // B
                            data[k + 3] = 255; // A
                        }

                        k = k + 4;
                    }
                }
            }
        }
    }

    // drawing and scaling
    _interactions_hidden_canvas.putImageData(imgData, 0, 0);
    _interactions_image_obj.onload = function () {
        ctx.canvas.width = _interactions_scale * _w;
        ctx.canvas.height = _interactions_scale * _h;
        ctx.scale(_interactions_scale, _interactions_scale);
        ctx.drawImage(_interactions_image_obj, 0, 0);

        for (let [pred, xy] of pred_map) {
            let r = Math.floor(xy / _w);
            let c = Math.floor(xy - r * _w);
            let label = _sup_label_map_inv[pred];
            if (label !== undefined && label.charAt(0) !== '_')
                draw_text(ctx, c, r + 11,11, _sup_label_map_inv[pred], "white");
        }

        _interactions_finished_drawing = true;

        if (_save_interaction_pngs) {
            ctx.canvas.toBlob(function (blob) {
                saveAs(blob, "interactions.png");
            });
        }
    };

    _interactions_image_obj.src = _interactions_hidden_canvas_html.toDataURL();
}

function get_and_draw_supervisions(frame) {
    log("Requested: sup.targets");
    _sup_targets = undefined;
    _sup_indices = undefined;

    var ctx = get_canvas("Supervisions", _w, _h); // ensure this is immediately returned (to gain a top position)

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/sup.targets",
        dataType: "arraybuffer",
        data: {
            frame: frame,
            sync: _sync_follow ? 1 : 0
        },
        success: function (array_buffer) {
            log("Returned: sup.targets");
            try {
                _sup_targets = getDataFromNumPyArray(array_buffer).data;
                console.log("sup_targets: " + (_sup_targets));

                if (_sup_indices !== undefined && _frame_pixels !== undefined) {
                    _draw_supervisions();
                }
            } catch (err) {
                handle_ajax_draw_error(ctx, false, true);
                return;
            }

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: sup.targets (error)");
            handle_ajax_draw_error(ctx, false, true);
        }
    }));

    _ajax_pending_requests++;
    _ajax_requests_that_did_not_failed++;

    _pending_ajax_calls.push($.ajax({
        type: "GET",
        url: "/sup.indices",
        dataType: "arraybuffer",
        data: {
            frame: frame,
            sync: _sync_follow ? 1 : 0
        },
        success: function (array_buffer) {
            log("Returned: sup.indices");
            try {
                _sup_indices = getDataFromNumPyArray(array_buffer).data;
                console.log("sup_indices: " + (_sup_indices));

                if (_sup_targets !== undefined && _frame_pixels !== undefined) {
                    _draw_supervisions();
                }
            } catch (err) {
                handle_ajax_draw_error(ctx, false, true);
                return;
            }

            _ajax_pending_requests--;
        },
        error: function () {
            log("Returned: sup.indices (error)");
            handle_ajax_draw_error(ctx, false, true);
        }
    }));
}

function _draw_supervisions()
{
    var ctx = get_canvas("Supervisions");
    var imgData = ctx.createImageData(_w, _h);
    var data = imgData.data;

    console.log("DRAWING SUP!");

    if (_frame_pixels === -1) {
        console.log("FAIL!!!!");
        drawNotFoundOrError(ctx, ctx.canvas.width, ctx.canvas.height);
        return;
    }

    copyAndGoGray(data, _frame_pixels);

    // mask
    var supervised = new Uint8Array(_w * _h + 2 * _w + 2); // init to zero (making it bigger to avoid border clamping)
    var offset = _w + 1;

    var k,i;
    var beta = 0.3;
    var color;
    var umbeta = 1.0 - beta;

    for (i = 0; i < _sup_targets.length; i = i + 1) {
        color = _colors[_sup_targets[i] % 10];
        k = _sup_indices[i] * 4;

        data[k] = beta * data[k] + (umbeta) * color[0]; // R
        data[k + 1] = beta * data[k + 1] + (umbeta) * color[1]; // G
        data[k + 2] = beta * data[k + 2] + (umbeta) * color[2]; // B
        data[k + 3] = 255; // A

        supervised[_sup_indices[i] + offset] = 1;
    }

    ctx.putImageData(imgData, 0, 0);

    // drawing crosses on isolated pixels
    var row, col;
    for (i = 0; i < _sup_targets.length; i = i + 1) {
        k = _sup_indices[i] + offset;

        if (supervised[k - 1] === 0
            && supervised[k + 1] === 0
            && supervised[k - _w] === 0
            && supervised[k - _w - 1] === 0
            && supervised[k - _w + 1] === 0
            && supervised[k + _w] === 0
            && supervised[k + _w - 1] === 0
            && supervised[k + _w + 1] === 0) { // sloppy border handling (faster - mistakes at the border, who cares?)
            color = _colors[_sup_targets[i] % 10];
            row = Math.floor(_sup_indices[i] / _w);
            col = _sup_indices[i] - row * _w;

            draw_cross(ctx, col, row, 8,
                '#' + color[0].toString(16) + color[1].toString(16) + color[2].toString(16), 5);
        }
    }
}