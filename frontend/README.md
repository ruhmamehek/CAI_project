# SEC Filings QA Frontend

Vue.js frontend for the SEC Filings Question-Answering system.

## Quick Start

### Option 1: Docker (Recommended)

1. **Start the frontend container:**
   ```bash
   docker compose up -d
   ```

2. **Access the application:**
   - Frontend: `http://localhost:3000`
   - Backend: `http://localhost:8000` (must be running separately)

3. **View logs:**
   ```bash
   docker compose logs -f
   ```

4. **Stop the container:**
   ```bash
   docker compose down
   ```

### Option 2: Local Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Access the application:**
   - Frontend: `http://localhost:8080` (Vite default port)
   - Backend: `http://localhost:8000` (must be running)

4. **Build for production:**
   ```bash
   npm run build
   ```

5. **Preview production build:**
   ```bash
   npm run preview
   ```

## Prerequisites

- **Backend Service**: The QA backend must be running on `http://localhost:8000`
  - See `../backend/qa/README.md` for backend setup instructions

## Configuration

### API URL

The frontend connects to the backend API. By default, it uses:
- `http://localhost:8000` (works for both Docker and local dev)

You can override this by setting the `VITE_API_URL` environment variable during build:
```bash
VITE_API_URL=http://your-backend-url:8000 npm run build
```

### Docker Configuration

The Docker setup:
- Builds the Vue.js app using `npm run build`
- Serves the production build using `npm run preview`
- Exposes the frontend on port `3000`
- Uses `extra_hosts` to access host's localhost (for backend communication)

## Project Structure

```
frontend/
├── src/
│   ├── App.vue           # Main application component
│   ├── main.js           # Application entry point
│   ├── assets/
│   │   └── style.css     # Global styles
│   └── components/
│       ├── QueryForm.vue # Query input form
│       └── Results.vue   # Results display
├── index.html            # HTML template
├── vite.config.js        # Vite configuration
├── package.json          # Dependencies and scripts
├── Dockerfile            # Docker build configuration
└── docker-compose.yml    # Docker Compose configuration
```

## Available Scripts

- `npm run dev` - Start development server with hot-reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

## Docker Commands

```bash
# Build and start
docker compose up -d

# Rebuild after changes
docker compose up --build -d

# View logs
docker compose logs -f frontend

# Stop
docker compose down

# Remove container and volumes
docker compose down -v
```

## Troubleshooting

### Backend Connection Issues

If the frontend can't connect to the backend:
1. Ensure the backend is running on `http://localhost:8000`
2. Check backend logs: `docker compose logs -f` (in backend/qa directory)
3. Verify backend health: `curl http://localhost:8000/health`

### Port Already in Use

If port 3000 is already in use:
- **Docker**: Change the port mapping in `docker-compose.yml`:
  ```yaml
  ports:
    - "3001:3000"  # Use port 3001 instead
  ```
- **Local**: Vite will automatically use the next available port

### CORS Errors

If you see CORS errors, ensure:
- Backend has CORS enabled (should be configured in `qa_service.py`)
- You're using the correct backend URL

## Development Notes

- The frontend uses Vue 3 with Composition API
- Styling uses a dark theme (see `src/assets/style.css`)
- API communication is handled in `App.vue` component
- The app automatically detects the environment and uses the appropriate API URL

