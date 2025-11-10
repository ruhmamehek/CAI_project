# SEC Filings QA Frontend

Vue.js 3 frontend for the SEC Filings QA System built with Vite.

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Backend service running (see `backend/qa/README.md`)

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:8080`

3. **Build for production:**
   ```bash
   npm run build
   ```
   The built files will be in the `dist` directory.

4. **Preview production build:**
   ```bash
   npm run preview
   ```

## Project Structure

```
frontend/
├── src/
│   ├── assets/
│   │   └── style.css          # Global styles
│   ├── components/
│   │   ├── QueryForm.vue      # Query form component
│   │   └── Results.vue        # Results display component
│   ├── App.vue                # Main app component
│   └── main.js                # Entry point
├── index.html                 # HTML template
├── vite.config.js            # Vite configuration
├── package.json              # Dependencies and scripts
└── README.md                 # This file
```

## Configuration

### API URL

The frontend connects to the backend API. By default, it's set to `http://localhost:8000`.

To change the API URL, edit `src/App.vue`:

```javascript
data() {
  return {
    apiUrl: 'http://localhost:8000'  // Change this to your backend URL
  }
}
```

### Vite Proxy

The `vite.config.js` includes a proxy configuration for development. If you want to use the proxy instead of direct API calls, you can modify the API calls to use `/api` prefix.

## Usage

1. **Enter your question** in the textarea
2. **Optionally set filters:**
   - **Ticker**: Filter by company (AAPL, MSFT, GOOGL)
   - **Year**: Filter by year (2022, 2023, 2024)
   - **Filing Type**: Filter by filing type (10-K, 10-Q)
   - **Top K**: Number of chunks to retrieve (default: 20)
3. **Click "Submit Query"**
4. **View results:**
   - **Answer**: Generated answer from the RAG system
   - **Sources**: List of source documents with metadata and relevance scores

## Development

### Adding New Components

1. Create a new `.vue` file in `src/components/`
2. Import and use it in `App.vue` or other components

Example:
```vue
<template>
  <MyComponent />
</template>

<script>
import MyComponent from './components/MyComponent.vue'

export default {
  components: {
    MyComponent
  }
}
</script>
```

### Styling

- Global styles are in `src/assets/style.css`
- Component-specific styles can be added in the `<style>` section of each `.vue` file
- Scoped styles can be used with `<style scoped>`

## API Integration

The frontend makes POST requests to:
- `POST /query` - Submit queries and get answers

Request format:
```json
{
  "query": "What was Apple's revenue in 2023?",
  "filters": {
    "ticker": "AAPL",
    "year": "2023"
  },
  "top_k": 20
}
```

Response format:
```json
{
  "answer": "...",
  "sources": [
    {
      "ticker": "AAPL",
      "filing_type": "10-K",
      "year": "2023",
      "accession_number": "...",
      "score": 0.85,
      "chunk_id": "..."
    }
  ],
  "num_chunks_retrieved": 5
}
```

## Troubleshooting

### CORS Errors
- CORS is enabled in the backend Flask service
- Verify backend is running on the correct port (default: 8000)
- Check that `flask-cors` is installed in the backend

### API Connection Errors
- Check that backend is running on the correct port
- Verify API URL in `src/App.vue`
- Check browser console for detailed errors

### Build Errors
- Ensure all dependencies are installed: `npm install`
- Check Node.js version (v16 or higher required)
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`

## Production Deployment

1. **Build the project:**
   ```bash
   npm run build
   ```

2. **Serve the `dist` directory:**
   - Use a static file server (nginx, Apache, etc.)
   - Or use a hosting service (Vercel, Netlify, etc.)

3. **Configure API URL:**
   - Update `apiUrl` in `src/App.vue` to your production API URL
   - Or use environment variables for configuration

## Technologies

- **Vue.js 3**: Progressive JavaScript framework
- **Vite**: Next-generation frontend build tool
- **JavaScript**: ES6+ features
- **CSS3**: Modern styling with gradients and animations
