document.addEventListener("DOMContentLoaded", function () {
    const checkNowButton = document.getElementById("check-now");
    const statusText = document.getElementById("status-text");
    const resultText = document.getElementById("result-text");

    checkNowButton.addEventListener("click", async function () {
        statusText.textContent = "Starting hardware check...";
        resultText.textContent = "";

        try {
            // Step 1: Request ESP32 to start measurement
            let espResponse = await fetch("http://192.168.4.1/start");  // Replace with your ESP32 IP
            let sensorData = await espResponse.json();  // Get the heart rate & temperature

            if (!sensorData.heartRate || !sensorData.temperature) {
                throw new Error("Invalid sensor readings from ESP32.");
            }

            console.log("Sensor Data from ESP32:", sensorData);
            statusText.textContent = "Processing data...";

            // Step 2: Send sensor data to ML model for diagnosis
            let mlResponse = await fetch("http://localhost:5000/predict", { // Replace with your ML API URL
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(sensorData)
            });

            let mlResult = await mlResponse.json();  // Get ML model output
            console.log("ML Prediction:", mlResult);

            // Step 3: Display result on the UI
            statusText.textContent = "Diagnosis complete.";
            resultText.textContent = `Health Status: ${mlResult.status}`;  // Display "Healthy" or "Unhealthy"

        } catch (error) {
            console.error("Error:", error);
            statusText.textContent = "Error: " + error.message;
        }
    });
});
