
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const { Pinecone } = require('@pinecone-database/pinecone');

// --- Configuration ---
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = 'bmw-z3-guide';
const PORT = process.env.PORT || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';

const app = express();

// --- Middleware ---
app.use(cors({
  origin: NODE_ENV === 'production' ? ['https://exp.host', 'https://expo.dev'] : '*',
  credentials: true
}));
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

let pinecone, pipe, guideData;

// --- Initialization ---
const initialize = async () => {
  try {
    if (PINECONE_API_KEY === 'YOUR_API_KEY') {
      throw new Error('Please set PINECONE_API_KEY environment variable');
    }
    
    console.log("Loading guide data...");
    const dataPath = './data.json';
    if (!fs.existsSync(dataPath)) {
      throw new Error('data.json file not found');
    }
    guideData = JSON.parse(fs.readFileSync(dataPath, 'utf-8')).bmw_z3_guide;
    console.log(`Loaded ${Object.keys(guideData).length} data categories`);

    console.log("Initializing Pinecone and pipeline...");
    pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
    
    // Use dynamic import for ESM module
    const { pipeline } = await import('@xenova/transformers');
    pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log("Initialization complete.");
  } catch (e) {
    console.error("Initialization failed:", e.message);
    // Don't exit in production, let the server start but return errors
    if (NODE_ENV !== 'production') {
      process.exit(1);
    }
  }
};

// --- Search Endpoint ---
app.post('/search', async (req, res) => {
  if (!pinecone || !pipe || !guideData) {
    return res.status(503).json({ 
      error: 'Server is not ready, please try again later.',
      details: !guideData ? 'Data not loaded' : 'AI services not initialized'
    });
  }

  const { query } = req.body;
  if (!query || typeof query !== 'string' || query.trim().length === 0) {
    return res.status(400).json({ error: 'Valid query string is required' });
  }

  try {
    // 1. Generate query embedding
    const queryEmbedding = await pipe(query, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(queryEmbedding.data);

    // 2. Query Pinecone
    const index = pinecone.index(PINECONE_INDEX_NAME);
    const response = await index.query({
      topK: 15, // Fetch more results to find unique guides
      vector: queryVector,
      includeMetadata: true,
    });

    // 3. Retrieve full guides from data.json
    const retrievedGuides = new Map();
    for (const match of response.matches) {
      const source = match.metadata.source;
      const pathParts = source.split(' > ');

      const category = pathParts[0];
      const guideKeyOrIndex = pathParts[1];

      let guide;
      let uniqueId;

      if (guideData[category]) {
        if (Array.isArray(guideData[category])) {
          const index = parseInt(guideKeyOrIndex);
          if (!isNaN(index) && guideData[category][index]) {
            guide = guideData[category][index];
            uniqueId = `${category}-${index}`;
          }
        } else {
          if (guideData[category][guideKeyOrIndex]) {
            guide = guideData[category][guideKeyOrIndex];
            uniqueId = guideKeyOrIndex;
          }
        }
      }

      if (guide && !retrievedGuides.has(uniqueId)) {
        retrievedGuides.set(uniqueId, {
          id: uniqueId,
          ...guide,
          score: match.score,
        });
      }
    }

    const results = Array.from(retrievedGuides.values());
    results.sort((a, b) => b.score - a.score); // Sort by relevance score

    res.json(results.slice(0, 5)); // Return top 5 unique guides

  } catch (e) {
    console.error("Search failed:", e);
    res.status(500).json({ 
      error: 'Failed to perform search.',
      details: NODE_ENV === 'development' ? e.message : undefined
    });
  }
});

// --- Data Endpoint ---
app.get('/data', async (req, res) => {
  if (!guideData) {
    return res.status(503).json({ error: 'Server is not ready, please try again later.' });
  }

  try {
    // Return all guide data for category navigation
    res.json({ bmw_z3_guide: guideData });
  } catch (e) {
    console.error("Data endpoint failed:", e);
    res.status(500).json({ 
      error: 'Failed to load data.',
      details: NODE_ENV === 'development' ? e.message : undefined
    });
  }
});

// --- Error handling middleware ---
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    details: NODE_ENV === 'development' ? err.message : undefined
  });
});

// --- 404 handler ---
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// --- Start Server ---
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT} in ${NODE_ENV} mode`);
  initialize();
});

// --- Graceful shutdown ---
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});
