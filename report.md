# Munder Difflin Multi-Agent System: A Comprehensive Report

## 1. Introduction

This report details the design, implementation, and evaluation of a multi-agent system developed for the Beaver's Choice Paper Company. The company faced significant operational challenges in managing inventory, responding to customer inquiries, and generating timely quotes, leading to potential revenue loss. The objective of this project was to create a robust, automated solution using a multi-agent architecture to streamline these processes.

The developed system leverages the `smolagents` framework to orchestrate a team of specialized agents that handle inventory management, customer quoting, and order finalization. By automating these core business functions, the system provides accurate, responsive, and reliable service, ensuring optimal stock levels and efficient transaction processing.

## 2. System Architecture and Design

The system is architected around a central orchestrator that manages a structured workflow, delegating tasks to a set of specialized worker agents. This design ensures a clear separation of concerns, making the system modular, scalable, and easier to maintain.
The final system architecture is the result of an iterative design process. The initial implementation utilized a different orchestration technique where the `OrchestratorAgent` was configured with a detailed system prompt and a list of `managed_agents`. In this model, the orchestrator was responsible for creating and executing a multi-step plan based on its prompt, delegating tasks to the quoting, inventory, and ordering agents sequentially. This approach effectively leveraged the agent's state and memory to manage the workflow.
However, this initial design proved to be more prone to errors. The complexity of the multi-step workflow meant the orchestrator could sometimes get confused, misinterpret the output from a sub-agent, or fail to follow the prescribed steps accurately. To enhance reliability, the architecture was refactored.
The current, more robust design replaces the complex prompt-driven workflow with a single, powerful tool called `handle_customer_request` within the `OrchestratorAgent`. This tool programmatically controls the entire process, calling the specialized agents in a fixed, reliable sequence. A key improvement was the introduction of the `AnalysisAgent`, which is tasked with interpreting the output from the `QuotingAgent` and returning a simple, machine-readable JSON object to dictate the next step. This isolates complex decision logic from the main orchestration flow, significantly reducing ambiguity and improving the system's overall consistency and accuracy. The code for the initial `managed_agents` approach remains commented out in `project_starter.py` for reference.

### 2.1. Agent Workflow Diagram Explanation

As illustrated in the `workflow_diagram.md` file, the system's workflow is initiated by a user request and follows a logical, sequential process:

1.  **User Request & Normalization**: A customer request is first processed by the `normalize_item_names` function. This crucial pre-processing step uses TF-IDF vector similarity to match ambiguous customer phrasing (e.g., "200 sheets of A4 glossy paper") to official inventory item names, significantly improving the accuracy of subsequent steps.
2.  **Orchestration**: The normalized request is passed to the `OrchestratorAgent`, which acts as the central controller for the entire process.
3.  **Quoting & Analysis**: The `OrchestratorAgent` invokes the `QuotingAgent` to generate a detailed quote, including price, stock availability, and delivery estimates. This quote is then passed to the `AnalysisAgent`, which parses the information and returns a structured JSON decision: `FINALIZE_ORDER`, `REORDER_STOCK`, or `CANNOT_FULFILL`.
4.  **Execution**: Based on the `AnalysisAgent`'s decision, the `OrchestratorAgent` proceeds with the appropriate action:
    *   If stock is sufficient, it calls the `InventoryAgent` to confirm stock and then the `OrderingAgent` to finalize the sale.
    *   If stock is insufficient but can be reordered in time, it calls the `InventoryAgent` to place a stock order. If successful, it proceeds to call the `OrderingAgent`.
    *   If the request cannot be fulfilled, the process terminates, and a message is prepared for the user.
5.  **Final Response**: The `OrchestratorAgent` synthesizes the results of the workflow into a clear, natural-language response for the user.

### 2.2. Agent Roles and Responsibilities

The system is composed of five distinct agents, each with a specialized role:

*   **`OrchestratorAgent`**: The brain of the operation. It does not perform business logic itself but manages the end-to-end workflow through its primary tool, `handle_customer_request`. It coordinates the other agents to ensure requests are handled consistently and according to business rules.
*   **`QuotingAgent`**: The customer-facing specialist. It generates quotes by using its tools to get pricing (`get_pricing_and_availability`), search past transactions for loyalty discount opportunities (`quote_history`), and apply a standard commission and any applicable discounts (`apply_commission_and_discount`).
*   **`InventoryAgent`**: The supply chain manager. It is responsible for all stock-related tasks, including checking levels (`check_stock_levels`), determining reorder needs (`check_reorder_status`), and placing purchase orders (`place_stock_order`). It also provides financial oversight by checking the company's cash balance (`check_cash_balance`) before authorizing a purchase.
*   **`OrderingAgent`**: The transaction processor. Its sole responsibility is to finalize a customer sale by creating a `sales` transaction in the database using its `finalize_order` tool, but only after confirming sufficient stock.
*   **`AnalysisAgent`**: A specialized decision-making agent. It is a tool-less agent that receives the structured output from the `QuotingAgent` and uses its analytical capabilities to determine the next logical step in the workflow, returning a simple, machine-readable JSON object to the Orchestrator. This isolates complex decision logic from the main orchestration flow, improving reliability.

## 3. Evaluation and Performance

The system was evaluated using the `quote_requests_sample.csv` dataset, and the results were logged in `test_results.csv`.

### 3.1. Strengths

*   **Successful Fulfillment**: The system successfully fulfilled multiple orders (e.g., requests 4, 5, 12), correctly processing transactions and updating the company's cash balance and inventory value as expected.
*   **Constraint Adherence**: The system correctly identified and handled requests that could not be fulfilled. As seen in requests 3, 6, and 9, it accurately determined when stock was insufficient and the delivery timeline could not be met, preventing impossible orders from being accepted.
*   **Structured Decision-Making**: The workflow demonstrated a consistent and logical decision-making process. For requests where stock was low, it correctly identified the need to reorder (e.g., requests 1, 7, 10) before attempting to finalize the sale.

### 3.2. Areas for Improvement

1.  **Robust Response Parsing**: The `test_results.csv` file shows that some final responses are raw JSON objects or error-like messages instead of polished, user-friendly text. This is due to the LLM occasionally failing to follow the final summarization instruction. The `OrchestratorAgent`'s final step could be enhanced with more robust parsing logic to catch these malformed outputs and reformat them into a consistent, natural-language response.
2.  **Consolidated Handling of Multi-Item Requests**: The current system processes items in a request sequentially. If a request contains multiple items and only one is out of stock, the entire process may fail or produce a confusing result. A future improvement would be to enhance the `OrchestratorAgent` to handle partial fulfillment, allowing it to inform the user which items are available and ask if they wish to proceed with a partial order.

## 4. Conclusion

The multi-agent system successfully automates the core inventory, quoting, and sales operations for the Beaver's Choice Paper Company. By employing a structured, orchestrated workflow with specialized agents, the system provides a reliable and efficient solution that meets all project requirements. While there are opportunities for improvement, particularly in response formatting and handling complex orders, the current implementation provides a strong foundation for revolutionizing the company's operations.
