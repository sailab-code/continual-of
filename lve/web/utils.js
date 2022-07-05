function getDataFromNumPyArray(buf) {
    var magic = String.fromCharCode.apply(null, new Uint8Array(buf.slice(0, 6)));
    if (magic.slice(1, 6) !== 'NUMPY') {
        throw new Error('Unknown file type, expected: NUMPY');
    }

    var version = new Uint8Array(buf.slice(6, 8));
    var view = new DataView(buf.slice(8, 10));
    var headerLength = view.getUint8(0);
    headerLength |= view.getUint8(1) << 8;
    var headerStr = String.fromCharCode.apply(null, new Uint8Array(buf.slice(10, 10 + headerLength)));
    var offsetBytes = 10 + headerLength;
    var info;
    eval("info = " + headerStr.toLowerCase().replace('(', '[').replace('),', ']'));

    var h = info.shape[0];
    var w = info.shape[1];
    var c = info.shape[2];
    var z = info.shape[3];
    var c_first = false;

    /*
    Here we perform a weird renaming of the array coordinates in function of some properties of the received numpy
    array.
    1) If we received an array with 3 dimensions, then we are in front of an "old-style array",
       and they are "h,w,c", respectively, with "c_first=false".
    2) If we received an array with 4 dimensions, then we are in from of a "new-style array" (i.e., pytorch-related).
       2a) If the first dimension is equal to 1, then such dimension is discarded, and the other array dimensions
           are "c,h,w", with "c_first=true"
       2b) If the first dimension is greater than 1, then we assume we are in front of a convolutional filter
           whose size is "features, input dims, kernel_size, kernel_size".
           We discard the last dimension and we name the other dimensions "h,w,c*c" with "c_first=false", that is what
           the drawing routine expects (the old-style array of filters was like this).
     */

    if (c === undefined) {
        c = 1;
    }

    if (z !== undefined) {
        if (h ===  1) {
            h = c;
            c = w;
            w = z;
            c_first = true;
        } else {
            c = c * z;
        }
    }

    var data;
    if (info.descr === "|u1") {
        data = new Uint8Array(buf, offsetBytes);
    } else if (info.descr === "|i1") {
        data = new Int8Array(buf, offsetBytes);
    } else if (info.descr === "<u2") {
        data = new Uint16Array(buf, offsetBytes);
    } else if (info.descr === "<i2") {
        data = new Int16Array(buf, offsetBytes);
    } else if (info.descr === "<u4") {
        data = new Uint32Array(buf, offsetBytes);
    } else if (info.descr === "<i4") {
        data = new Int32Array(buf, offsetBytes);
    } else if (info.descr === "<f4") {
        data = new Float32Array(buf, offsetBytes);
    } else if (info.descr === "<f8") {
        data = new Float64Array(buf, offsetBytes);
    } else if (info.descr === "<i8") {
        data = new BigUint64Array(buf, offsetBytes);
    } else {
        throw new Error('Unknown numeric dtype!')
    }

    return {data: data, w: w, h: h, c: c, c_first: c_first}
}

function copyAndGoGray(dest_pixels, source_pixels) {
    var g;
    for (var i = 0; i < source_pixels.length; i = i + 4) {
        g = source_pixels[i] * .3 + source_pixels[i + 1] * .59 + source_pixels[i + 2] * .11;
        dest_pixels[i] = g; // R
        dest_pixels[i + 1] = g; // G
        dest_pixels[i + 2] = g; // B
        dest_pixels[i + 3] = source_pixels[i + 3]; // A
    }
}

function goBlack(pixels) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i] = 0; // R
        pixels[i + 1] = 0; // G
        pixels[i + 2] = 0; // B
        pixels[i + 3] = 255; // A
    }
}

function goWhite(pixels) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i] = 255; // R
        pixels[i + 1] = 255; // G
        pixels[i + 2] = 255; // B
        pixels[i + 3] = 255; // A
    }
}

function goRed(pixels) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i] = 255; // R
        pixels[i + 1] = 0; // G
        pixels[i + 2] = 0; // B
        pixels[i + 3] = 255; // A
    }
}

function copyPixels(dest_pixels, source_pixels) {
    for (var i = 0; i < dest_pixels.length; i = i + 4) {
        dest_pixels[i] = source_pixels[i]; // R
        dest_pixels[i + 1] = source_pixels[i + 1]; // G
        dest_pixels[i + 2] = source_pixels[i + 2]; // B
        dest_pixels[i + 3] = source_pixels[i + 3]; // A
    }
}

function scalePix(pixels, rho) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i] = rho * pixels[i]; // R
        pixels[i + 1] = rho * pixels[i + 1]; // G
        pixels[i + 2] = rho * pixels[i + 2]; // B
    }
}

function drawNotFoundOrError(ctx, w, h) {
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.fillStyle = "black";
    ctx.fill();

    ctx.font = "14px Arial";
    ctx.fillStyle = "white";
    ctx.fillText("Not Found!", 10, 20);
}

function scale_canvas(cavas_html, ctx, w, h) {
    //if (cavas_html.width < w || cavas_html.height < h) {
        var imageObject = new Image();
        imageObject.onload = function () {
            var sw = w / cavas_html.width;
            var sh = h / cavas_html.height;
            cavas_html.width = sw;
            cavas_html.height = sh;
            ctx.scale(2, 2);
            ctx.drawImage(imageObject, 0, 0);
        };
        imageObject.src = cavas_html.toDataURL();
    //}
}

function hsvToRgb(h, s, v) {
    var r, g, b;

    var i = Math.floor(h * 6);
    var f = h * 6 - i;
    var p = v * (1 - s);
    var q = v * (1 - f * s);
    var t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0:
            r = v; g = t; b = p;
            break;
        case 1:
            r = q; g = v; b = p;
            break;
        case 2:
            r = p; g = v; b = t;
            break;
        case 3:
            r = p; g = q; b = v;
            break;
        case 4:
            r = t; g = p; b = v;
            break;
        case 5:
            r = v; g = p; b = q;
            break;
    }

    return [r * 255, g * 255, b * 255];
}

function generate_array_of_colors() {
    var colors = [];
    colors[0] = [];
    colors[0][0] = 230.0;
    colors[0][1] = 25.0;
    colors[0][2] = 75.0;
    colors[1] = [];
    colors[1][0] = 60.0;
    colors[1][1] = 180.0;
    colors[1][2] = 75.0;
    colors[2] = [];
    colors[2][0] = 255.0;
    colors[2][1] = 225.0;
    colors[2][2] = 25.0;
    colors[3] = [];
    colors[3][0] = 0.0;
    colors[3][1] = 130.0;
    colors[3][2] = 200.0;
    colors[4] = [];
    colors[4][0] = 245.0;
    colors[4][1] = 130.0;
    colors[4][2] = 48.0;
    colors[5] = [];
    colors[5][0] = 145.0;
    colors[5][1] = 30.0;
    colors[5][2] = 180.0;
    colors[6] = [];
    colors[6][0] = 70.0;
    colors[6][1] = 240.0;
    colors[6][2] = 240.0;
    colors[7] = [];
    colors[7][0] = 240.0;
    colors[7][1] = 50.0;
    colors[7][2] = 230.0;
    colors[8] = [];
    colors[8][0] = 210.0;
    colors[8][1] = 245.0;
    colors[8][2] = 60.0;
    colors[9] = [];
    colors[9][0] = 250.0;
    colors[9][1] = 190.0;
    colors[9][2] = 190.0;
    return colors;
}

function draw_cross(ctx, x, y, len, color, tickness) {
    ctx.lineWidth = tickness;
    ctx.strokeStyle = '#000000';
    ctx.beginPath();
    var half_cross_len = len / 2;
    ctx.moveTo(x - half_cross_len, y);
    ctx.lineTo(x + half_cross_len, y);
    ctx.moveTo(x, y - half_cross_len);
    ctx.lineTo(x, y + half_cross_len);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
}

function draw_circle(ctx, x, y, radius, color, tickness) {
    ctx.lineWidth = tickness;
    ctx.strokeStyle = '#000000';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
}

function draw_rect(ctx, x, y, w, h, color, tickness) {
    ctx.lineWidth = tickness;
    ctx.strokeStyle = '#000000';
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
}

function draw_text(ctx, x, y, size, text, color) {
    ctx.font = size + "px Arial";
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
}

function draw_text_bold(ctx, x, y, size, text, color) {
    ctx.font = "bold " + size + "px Arial";
    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
}