(function (exports) {

  exports.URL = exports.URL || exports.webkitURL;

  exports.requestAnimationFrame = exports.requestAnimationFrame ||
    exports.webkitRequestAnimationFrame || exports.mozRequestAnimationFrame ||
    exports.msRequestAnimationFrame || exports.oRequestAnimationFrame;

  exports.cancelAnimationFrame = exports.cancelAnimationFrame ||
    exports.webkitCancelAnimationFrame || exports.mozCancelAnimationFrame ||
    exports.msCancelAnimationFrame || exports.oCancelAnimationFrame;

  navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;

  var ORIGINAL_DOC_TITLE = document.title;
  var video = $('video');
  var canvas = document.createElement('canvas'); // offscreen canvas.
  var rafId = null;
  var setInt = null;
  var startTime = null;
  var endTime = null;

  namespace = '/live'; // no way

  // the socket.io documentation recommends sending an explicit package upon connection
  // this is specially important when using the global namespace
  console.log('http://' + document.domain + ':' + location.port + namespace);
  var socket = io.connect('http://' + document.domain + ':' + location.port + namespace);

  socket.on('connect', function () {
    console.log('connection event');
    socket.emit('event', { data: 'Client, Here' });
  });
  socket.on('response', function ({ data }) {

    // if (data.smile && data.smile)
    console.log('data', data);
    document.getElementById("img").setAttribute(
      'src', `data:image/jpeg;base64, ${data.out}`
     ,'width',`100px`, 'heigth',`100px` );
  });

  function $(selector) {
    return document.querySelector(selector) || null;
  }

  function toggleActivateRecordButton() {
    var b = $('#record-me');
    b.textContent = b.disabled ? 'Go live' : 'Streaming...';
    b.classList.toggle('recording');
    b.disabled = !b.disabled;
  }

  function turnOnCamera(e) {
    e.target.disabled = true;
    $('#record-me').disabled = false;

    video.controls = false;

    var finishVideoSetup_ = function () {
      // Note: video.onloadedmetadata doesn't fire in Chrome when using getUserMedia so
      // we have to use setTimeout. See crbug.com/110938.
      setTimeout(function () {
        video.width = 320;//video.clientWidth;
        video.height = 240;// video.clientHeight;
        // Canvas is 1/2 for performance. Otherwise, getImageData() readback is
        // awful 100ms+ as 640x480.
        canvas.width = video.width;
        canvas.height = video.height;
      }, 1000);
    };

    navigator.getUserMedia({ video: true, audio: false }, function (stream) {
      video.srcObject = stream;
      finishVideoSetup_();
    }, function (e) {
      alert('Fine, you get a movie instead of your beautiful face ;)');
      video.src = ''; // Nope, you get nothing
      finishVideoSetup_();
    });
  };

  function record() {
    /* Go live */
    var elapsedTime = $('#elasped-time');
    var ctx = canvas.getContext('2d');
    var CANVAS_HEIGHT = canvas.height;
    var CANVAS_WIDTH = canvas.width;

    startTime = Date.now();
    socket.emit('event', { data: 'RECORDING!' });
    toggleActivateRecordButton();
    $('#stop-me').disabled = false;


    function sendVideoFrame_() {
      // draw the video contents into the canvas x, y, width, height
      ctx.drawImage(video, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      // get the image data from the canvas object
      // and send them through websockets
      socket.emit('livevideo', { data: canvas.toDataURL('image/jpeg', 0.7) });  // Send video frame to server
      //console.log(canvas.toDataURL());
      document.title = 'Live streaming...' + Math.round((Date.now() - startTime) / 1000) + 's';
    };

    setInt = setInterval(function () { sendVideoFrame_() }, 1000 / 20);

  };

  function stop() {
    // cancelAnimationFrame(rafId);
    clearInterval(setInt);
    endTime = Date.now();
    $('#stop-me').disabled = true;
    document.title = ORIGINAL_DOC_TITLE;
    toggleActivateRecordButton();
  };

  function initEvents() {
    $('#camera-me').addEventListener('click', turnOnCamera);
    $('#record-me').addEventListener('click', record);
    $('#stop-me').addEventListener('click', stop);
  };

  initEvents();

  exports.$ = $;

})(window);
