const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const app = express();
const PORT = 3000;

app.use(cors());
// Serve static files (e.g., HTML, CSS, JS)
app.use(express.static(path.join(__dirname)));

// Endpoint to fetch the file list
app.get('/file-list', (req, res) => {
  const directoryPath = path.join(__dirname, 'circuit_descriptions');
  
  fs.readdir(directoryPath, (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      res.status(500).send('Unable to fetch file list');
      return;
    }
    res.json(files);
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
