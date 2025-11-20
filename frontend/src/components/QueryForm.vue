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
        <div class="auto-filter-option">
          <label>
            <input type="checkbox" v-model="auto_determine_filters" />
            Auto-determine filters from query
          </label>
        </div>
        <div class="filter-row">
          <div class="form-group">
            <label for="ticker">Ticker:</label>
            <select id="ticker" v-model="filters.ticker" :disabled="auto_determine_filters">
              <option value="">All Companies</option>
              <option value="JPM">JPM (JPMorgan Chase)</option>
              <option value="JNJ">JNJ (Johnson & Johnson)</option>
              <option value="MSFT">MSFT (Microsoft)</option>
              <option value="TSLA">TSLA (Tesla)</option>
              <option value="WMT">WMT (Walmart)</option>
              <option value="XOM">XOM (Exxon Mobil)</option>
              <option value="AAPL">AAPL (Apple)</option>
              <option value="GOOGL">GOOGL (Google)</option>
            </select>
          </div>

          <div class="form-group">
            <label for="year">Year:</label>
            <select id="year" v-model="filters.year" :disabled="auto_determine_filters">
              <option value="">All Years</option>
              <option value="2023">2023</option>
              <option value="2024">2024</option>
            </select>
          </div>

          <div class="form-group">
            <label for="item_number">Item Number:</label>
            <select id="item_number" v-model="filters.item_number" :disabled="auto_determine_filters">
              <option value="">All Items</option>
              <option value="1">Item 1 - Business</option>
              <option value="1A">Item 1A - Risk Factors</option>
              <option value="1B">Item 1B - Unresolved Staff Comments</option>
              <option value="1C">Item 1C - Cybersecurity</option>
              <option value="2">Item 2 - Properties</option>
              <option value="3">Item 3 - Legal Proceedings</option>
              <option value="4">Item 4 - Mine Safety Disclosures</option>
              <option value="5">Item 5 - Market for Registrant's Common Equity</option>
              <option value="6">Item 6 - Reserved</option>
              <option value="7">Item 7 - Management's Discussion and Analysis</option>
              <option value="7A">Item 7A - Quantitative and Qualitative Disclosures About Market Risk</option>
              <option value="8">Item 8 - Financial Statements and Supplementary Data</option>
              <option value="9">Item 9 - Changes in and Disagreements with Accountants</option>
              <option value="9A">Item 9A - Controls and Procedures</option>
              <option value="9B">Item 9B - Other Information</option>
              <option value="9C">Item 9C - Disclosure Regarding Foreign Jurisdictions</option>
              <option value="10">Item 10 - Directors, Executive Officers and Corporate Governance</option>
              <option value="11">Item 11 - Executive Compensation</option>
              <option value="12">Item 12 - Security Ownership of Certain Beneficial Owners</option>
              <option value="13">Item 13 - Certain Relationships and Related Transactions</option>
              <option value="14">Item 14 - Principal Accountant Fees and Services</option>
              <option value="15">Item 15 - Exhibit and Financial Statement Schedules</option>
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
        year: '',
        item_number: ''
      },
      top_k: 20,
      auto_determine_filters: true  // Default to auto-determine
    }
  },
  methods: {
    handleSubmit() {
      const payload = {
        query: this.query.trim(),
        auto_determine_filters: this.auto_determine_filters
      }

      // Add filters if any are selected (only if auto_determine_filters is false)
      if (!this.auto_determine_filters) {
        const activeFilters = {}
        if (this.filters.ticker) {
          activeFilters.ticker = this.filters.ticker
        }
        if (this.filters.year) {
          activeFilters.year = this.filters.year
        }
        if (this.filters.item_number) {
          activeFilters.item_number = this.filters.item_number
        }

        if (Object.keys(activeFilters).length > 0) {
          payload.filters = activeFilters
        }
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
        year: '',
        item_number: ''
      }
      this.top_k = 20
      this.auto_determine_filters = true
    }
  }
}
</script>

