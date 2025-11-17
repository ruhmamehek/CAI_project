<template>
  <section class="query-section">
    <form @submit.prevent="handleSubmit" class="query-form">
      <div class="form-group">
        <label for="query">Your Question:</label>
        <textarea
          id="query"
          v-model="query"
          placeholder="e.g., What was Apple's revenue in 2023?"
          rows="3"
          required
        ></textarea>
      </div>

      <div class="filters">
        <h3>Filters (Optional)</h3>
        <div class="filter-row">
          <div class="form-group">
            <label for="ticker">Ticker:</label>
            <select id="ticker" v-model="filters.ticker">
              <option value="">All Companies</option>
              <option value="JNJ">JNJ (Johnson & Johnson)</option>
              <option value="JPM">JPM (JPMorgan Chase)</option>
              <option value="MSFT">MSFT (Microsoft)</option>
              <option value="WMT">WMT (Walmart)</option>
              <option value="XOM">XOM (Exxon Mobil)</option>
            </select>
          </div>

          <div class="form-group">
            <label for="year">Year:</label>
            <select id="year" v-model="filters.year">
              <option value="">All Years</option>
              <option value="2023">2023</option>
              <option value="2024">2024</option>
            </select>
          </div>

          <div class="form-group">
            <label for="top_k">Top K:</label>
            <input
              type="number"
              id="top_k"
              v-model.number="top_k"
              min="1"
              max="50"
              placeholder="20"
            />
          </div>
        </div>
      </div>

      <div class="button-group">
        <button type="submit" :disabled="loading" class="submit-btn">
          <span v-if="loading">Processing...</span>
          <span v-else>Submit Query</span>
        </button>
        <button type="button" @click="clearForm" class="clear-btn" :disabled="loading">
          Clear
        </button>
      </div>
    </form>
  </section>
</template>

<script>
export default {
  name: 'QueryForm',
  props: {
    loading: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      query: '',
      filters: {
        ticker: '',
        year: ''
      },
      top_k: 20
    }
  },
  methods: {
    handleSubmit() {
      const payload = {
        query: this.query.trim()
      }

      const activeFilters = {}
      if (this.filters.ticker) {
        activeFilters.ticker = this.filters.ticker
      }
      if (this.filters.year) {
        activeFilters.year = this.filters.year
      }

      if (Object.keys(activeFilters).length > 0) {
        payload.filters = activeFilters
      }

      if (this.top_k) {
        payload.top_k = this.top_k
      }

      this.$emit('query-submitted', payload)
    },
    clearForm() {
      this.query = ''
      this.filters = {
        ticker: '',
        year: ''
      }
      this.top_k = 20
    }
  }
}
</script>

