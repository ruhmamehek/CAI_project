<template>
  <div id="app">
    <div class="container">
      <header>
        <h1>SEC Filings QA System</h1>
        <p class="subtitle">Ask questions about SEC filings using RAG</p>
      </header>

      <main>
        <QueryForm 
          @query-submitted="handleQuery"
          :loading="loading"
        />

        <div v-if="error" class="error-message">
          <strong>Error:</strong> {{ error }}
          <button @click="error = null" class="close-error">×</button>
        </div>

        <Results 
          v-if="result"
          :result="result"
        />

        <div v-if="loading" class="loading">
          <div class="spinner"></div>
          <p>Processing your query...</p>
        </div>
      </main>

      <footer>
        <p>SEC Filings QA System - RAG-powered question answering</p>
      </footer>
    </div>
  </div>
</template>

<script>
import QueryForm from './components/QueryForm.vue'
import Results from './components/Results.vue'

export default {
  name: 'App',
  components: {
    QueryForm,
    Results
  },
  data() {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    return {
      loading: false,
      error: null,
      result: null,
      apiUrl: apiUrl
    }
  },
  methods: {
    async handleQuery(payload) {
      this.error = null
      this.result = null
      this.loading = true

      try {
        const response = await fetch(`${this.apiUrl}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        this.result = data
      } catch (error) {
        console.error('Error submitting query:', error)
        this.error = error.message || 'An error occurred while processing your query.'
      } finally {
        this.loading = false
      }
    }
  },
  mounted() {
    fetch(`${this.apiUrl}/health`)
      .then(response => response.json())
      .then(data => {
        console.log('API health check:', data)
      })
      .catch(error => {
        console.warn('API health check failed:', error)
      })
  }
}
</script>

