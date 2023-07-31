const BASE_URL = 'https://iv000vo6mf.execute-api.ap-south-1.amazonaws.com/dev';

// Function to handle image upload and processing
document.getElementById("file").onchange = function (e) {
    const image = e.target.files[0];
    window.loadImage(image, function (img) {
        if (img.type === "error") {
            console.log("Couldn't load image:", img);
        } else {
            window.EXIF.getData(image, function () {
                console.log("Load image done!");
                const orientation = window.EXIF.getTag(this, "Orientation");
                const canvas = window.loadImage.scale(img, {
                    orientation: orientation || 0,
                    canvas: true,
                    maxWidth: 500,
                    maxHeight: 500,
                });
                $("#display").attr("src", "");
                $("#imagebox").attr("src", canvas.toDataURL());
            });
        }
    });
};

// Function to redirect to endpoint using BASE_URL
function tryIt(endpoint) {
    const url = BASE_URL + endpoint;
    window.location.href = url;
}