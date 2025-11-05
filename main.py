import ollama
import re
import matplotlib.pyplot as plt

class ConversationContext:
    """
    Manages conversation history for maintaining context with Ollama models.
    """
    def __init__(self, model="llama3.2:3b", system_prompt=None, reinforce_every=None):
        self.model = model
        self.messages = []
        self.system_prompt = system_prompt
        self.reinforce_every = reinforce_every
        self.turn_count = 0

        # Add system prompt if provided
        if system_prompt:
            self.messages.append({
                'role': 'system',
                'content': system_prompt
            })

    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.messages.append({
            'role': role,
            'content': content
        })

    def chat(self, user_message, stream=False):
        """
        Send a message and get a response while maintaining context.

        Args:
            user_message (str): The user's message
            stream (bool): Whether to stream the response

        Returns:
            str or generator: The assistant's response
        """
        # Increment turn counter
        self.turn_count += 1

        # Reinforce system prompt if needed
        if self.reinforce_every and self.turn_count % self.reinforce_every == 0 and self.system_prompt:
            user_message = user_message + '\n\n[System Reminder]\n' + self.system_prompt

        # Add user message to history
        self.add_message('user', user_message)

        if stream:
            return self._stream_chat()
        else:
            return self._chat()

    def _chat(self):
        """Non-streaming chat."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages
            )
            assistant_message = response['message']['content']

            # Add assistant response to history
            self.add_message('assistant', assistant_message)

            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"

    def _stream_chat(self):
        """Streaming chat."""
        try:
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True
            )

            # Collect the full response while streaming
            full_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                yield content

            # Add assistant response to history after streaming completes
            self.add_message('assistant', full_response)

        except Exception as e:
            yield f"Error: {str(e)}"

    def clear_history(self, keep_system=True):
        """
        Clear conversation history.

        Args:
            keep_system (bool): Whether to keep the system prompt
        """
        if keep_system and self.messages and self.messages[0]['role'] == 'system':
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []

    def get_history(self):
        """Get the full conversation history."""
        return self.messages


if __name__ == '__main__':
    # Example: Feed transcript
    print("Example: Debate Transcript")
    print("-" * 50)
    with open('Transcripts/Debate_transcript.txt', 'r') as f:
        transcript = f.read()

    # Split while keeping the speaker names
    parts = re.split(r'(\n)', transcript)

    # Filter for only parts that contain KENNEDY or NIXON
    names=['KENNEDY','NIXON','SMITH','FLEMING','NOVINS','WARREN','VANOCUR']
    speaker_segments = [p for p in parts if any(name in p for name in names)]

    speaker_response_true = [('KENNEDY' in p or 'NIXON' in p) for p in speaker_segments]
    speaker_response_pred = []

    # Load system prompt from file
    with open('systemrules.txt', 'r') as f:
        system_prompt = f.read()

    conversation = ConversationContext(
        #model="llama3.2:3b",
        model="deepseek-r1:8b",
        system_prompt=system_prompt,
        reinforce_every=1  # Reinforce system prompt every 5 turns
    )

    for segment in speaker_segments:
        print("Prompt: \n" + segment)
        print("\n \n \n Response:")

        # Collect all chunks into one complete response
        chuckychunk = ""
        for chunk in conversation.chat(segment):
            chuckychunk += chunk

        # Check if response contains "No response is permitted"
        if "No response is permitted" in chuckychunk:
            speaker_response_pred.append(False)
        else:
            speaker_response_pred.append(True)

        print(chuckychunk)
        print("\n \n \n")


    # Create boolean vector comparing true to pred
    comparison_vector = [true == pred for true, pred in zip(speaker_response_true, speaker_response_pred)]
    comparison_vector_number = [true - pred for true, pred in zip(speaker_response_true, speaker_response_pred)]


    # Calculate accuracy by comparing predictions to true labels
    correct_predictions = sum(comparison_vector)
    total_predictions = len(speaker_response_true)
    rules_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("="*50)
    print(f"Rules Accuracy: {rules_accuracy:.2%}")
    print(f"Correct: {correct_predictions}/{total_predictions}")
    print("="*50)

    # Plot comparison vector
    plt.figure(figsize=(12, 4))
    indices = range(len(comparison_vector))
    colors = ['green' if correct else 'red' for correct in comparison_vector]

    plt.bar(indices, [1 if c else 0 for c in comparison_vector], color=colors, alpha=0.7)
    plt.xlabel('Segment Index')
    plt.ylabel('Correct (1) / Incorrect (0)')
    plt.title(f'Prediction Correctness by Segment (Accuracy: {rules_accuracy:.2%})')
    plt.ylim(-0.1, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot comparison vector number
    plt.figure(figsize=(12, 4))
    plt.plot(indices, comparison_vector_number)
    plt.xlabel('Segment Index')
    plt.ylabel('0 correct, -1 False positive, +1 False negative')
    plt.title(f'Prediction Correctness by Segment (Accuracy: {rules_accuracy:.2%})')
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

