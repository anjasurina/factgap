from typing import Optional
from origins.custom_classes import InferenceTask, Phase, PromptType, lookup_prompt_type
from omegaconf import DictConfig
from origins.custom_classes import _prompt_category, GenerationPrompt, VerificationPrompt
from origins.prompts.prompt_utils import randomize_mc_options, render_j2_template, parse_text_in_tags
from origins.logging.print import log_with_color

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MC_ANSWER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def generate_prompts_with_template(
    inference_tasks: list[InferenceTask],
    cfg: DictConfig,
    include_reasoning: bool = True,
    allow_unsure: bool = False,
    base_model_prompt: bool = False,
) -> list[_prompt_category]:
    """
    Generate prompts for the model using a template.

    Args:
        inference_tasks (list[InferenceTask]): List of inference tasks
        cfg (DictConfig): Hydra config
        include_reasoning (bool): Whether to include reasoning in the prompts
        allow_unsure (bool): Whether to allow the model to respond with "Unsure"
        base_model_prompt (bool): Whether to use the base model prompt that appends "Response:" to the end of the prompt.

    Returns:
        list[_prompt_category]: List of generated prompts

    Raises:
        ValueError: If no prompts are generated
    """

    prompts = []
    for inference_task in inference_tasks:
        correct_problems_versions = inference_task.correct_problems_versions
        for cor_p in inference_task.correct_problems:
            if (
                correct_problems_versions
                and cor_p.version not in correct_problems_versions
            ):
                continue
            for template_name in cfg.infer.generation_template_names:
                generation_prompt = get_generation_prompt(
                    problem=cor_p.problem,
                    answer=cor_p.answer,
                    options=inference_task.control_answers[
                        : cfg.infer.num_mc_options - 1
                    ],
                    is_correct=True,
                    include_reasoning=include_reasoning,
                    template_name=template_name,
                    task_id=inference_task.task_id,
                    version=cor_p.version,
                    phase=inference_task.phase,
                    base_model_prompt=base_model_prompt,
                )
                prompts.append(generation_prompt)

            for template_name in cfg.infer.verification_template_names:
                verification_prompt_correct = get_double_critic_prompt(
                    problem=cor_p.problem,
                    answer=cor_p.answer,
                    correct_answer=cor_p.answer,
                    options=inference_task.control_answers[
                        : cfg.infer.num_mc_options - 1
                    ],
                    is_correct=True,
                    include_reasoning=include_reasoning,
                    eval_correct=True,
                    template_name=template_name,
                    task_id=inference_task.task_id,
                    version=cor_p.version,
                    phase=inference_task.phase,
                    allow_unsure=allow_unsure,
                    base_model_prompt=base_model_prompt,
                )
                verification_prompt_incorrect = get_double_critic_prompt(
                    problem=cor_p.problem,
                    answer=cor_p.answer,
                    correct_answer=cor_p.answer,
                    options=inference_task.control_answers[
                        : cfg.infer.num_mc_options - 1
                    ],
                    is_correct=True,
                    include_reasoning=include_reasoning,
                    eval_correct=False,
                    template_name=template_name,
                    task_id=inference_task.task_id,
                    version=cor_p.version,
                    phase=inference_task.phase,
                    allow_unsure=allow_unsure,
                    base_model_prompt=base_model_prompt,
                )
                prompts.append(verification_prompt_correct)
                prompts.append(verification_prompt_incorrect)

            if inference_task.control_answers:
                for i, control_answer in enumerate(inference_task.control_answers):
                    # Check if the index is within the bounds of the defined control answers
                    if i >= cfg.infer.num_control_answers:
                        break

                    # Exclude the currently selected control_answer from the options, then pick (num_mc_options - 1)
                    other_options = [
                        opt
                        for opt in inference_task.control_answers
                        if opt != control_answer
                    ][: cfg.infer.num_mc_options - 1]

                    for template_name in cfg.infer.verification_control_template_names:
                        prompts.append(
                            get_double_critic_prompt(
                                problem=cor_p.problem,
                                answer=control_answer,
                                correct_answer=cor_p.answer,
                                options=other_options,
                                is_correct=False,
                                include_reasoning=include_reasoning,
                                eval_correct=True,
                                template_name=template_name,
                                task_id=inference_task.task_id,
                                version=cor_p.version,
                                phase=inference_task.phase,
                                allow_unsure=allow_unsure,
                                base_model_prompt=base_model_prompt,
                            )
                        )
                        prompts.append(
                            get_double_critic_prompt(
                                problem=cor_p.problem,
                                answer=control_answer,
                                correct_answer=cor_p.answer,
                                options=other_options,
                                is_correct=False,
                                include_reasoning=include_reasoning,
                                eval_correct=False,
                                template_name=template_name,
                                task_id=inference_task.task_id,
                                version=cor_p.version,
                                phase=inference_task.phase,
                                allow_unsure=allow_unsure,
                                base_model_prompt=base_model_prompt,
                            )
                        )

        if inference_task.control_problems:
            control_problem_versions = inference_task.control_problems_versions
            for ctrl_p in inference_task.control_problems:
                if (
                    control_problem_versions
                    and ctrl_p.version not in control_problem_versions
                ):
                    continue

                for template_name in cfg.infer.generation_control_template_names:
                    prompts.append(
                        get_generation_prompt(
                            problem=ctrl_p.problem,
                            answer=ctrl_p.answer,
                            is_correct=False,
                            include_reasoning=include_reasoning,
                            template_name=template_name,
                            task_id=inference_task.task_id,
                            version=ctrl_p.version,
                            phase=inference_task.phase,
                            base_model_prompt=base_model_prompt,
                            #allow_unsure=allow_unsure,
                        )
                    )

    if not prompts:
        raise ValueError("No prompts generated. Check the inference tasks.")
    logger.debug(f"Generated {len(prompts)} prompts.")

    return prompts

def get_generation_prompt(
    problem: str,
    is_correct: bool,
    answer: Optional[str] = None,
    options: Optional[list[str]] = None,
    include_reasoning: bool = False,
    #allow_unsure: bool = False,
    base_model_prompt: bool = False,
    task_id: Optional[str] = None,
    version: Optional[str | int] = None,
    template_name: str = "generative_response.j2",
    phase: Phase = Phase.LEARN,
    ) -> GenerationPrompt:
    """
    Generates a prompt for a generative model based on the problem statement.

    Args:
        problem: The problem statement to be included in the prompt.
        is_correct: A boolean indicating whether the problem is correct.
        answer: The correct answer to the problem (if applicable).
        options: A list of options for multiple-choice questions (if applicable).
        include_reasoning: Whether to include reasoning in the response.
        base_model_prompt: Whether to use the base model prompt that appends "Response:" to the end of the prompt.
        task_id: An optional task identifier.
        version: An optional version identifier for the prompt.
        template_name: The name of the Jinja2 template file to use for rendering.

    Returns:
        A formatted string containing the prompt.
    """
    
    data = {
        "problem": problem,
        "include_reasoning": include_reasoning,
        "option_letters": MC_ANSWER_LETTERS,
        #"allow_unsure": allow_unsure,
        "base_model_prompt": base_model_prompt,
    }
    
    if lookup_prompt_type(template_name) == PromptType.GENERATIVE_MC:
        if not options or not answer:
            raise ValueError("For MC prompts, both 'options' and 'answer' must be provided.")
        answers, correct_index = randomize_mc_options(answer=answer, 
                                                      options=options)
        correct_answer_letter = MC_ANSWER_LETTERS[correct_index]        
    else:
        answers = []
        correct_answer_letter = None
    
    data["answers"] = answers
    
    prompt_text = render_j2_template(
        data={**data, "add_instructions": True},
        template_name=template_name
    )
    problem_statement = render_j2_template(
        data={**data, "add_instructions": False},
        template_name=template_name
    )
    
    return GenerationPrompt(
        prompt_text=prompt_text,
        is_correct=is_correct,
        template_name=template_name,
        correct_answer=answer,
        correct_answer_letter=correct_answer_letter,
        include_reasoning=include_reasoning,
        task_id=task_id,
        problem_statement=problem_statement,
        prompt_type=lookup_prompt_type(template_name),
        version=version, 
        phase=phase,
        #allow_unsure=allow_unsure,
        base_model_prompt=base_model_prompt,
    )
    
    
def get_double_critic_prompt(
    problem: str,
    answer: str,
    correct_answer: str,
    is_correct: bool,
    eval_correct: bool, 
    options: Optional[list[str]] = None,
    include_reasoning: bool = False,
    allow_unsure: bool = False,
    base_model_prompt: bool = False,
    task_id: Optional[str] = None,
    version: Optional[str | int] = None,
    template_name: str = "double_critic.j2",
    phase: Phase = Phase.LEARN,
    ) -> VerificationPrompt:
    """
    Generates a prompt for the double critic model.
    
    Args:
        problem (str): The problem statement.
        answer (str): The answer to the problem.
        is_correct (bool): Whether the answer is correct.
        eval_correct (bool): Whether to evaluate the (in)correctness of the answer.
        include_reasoning (bool): Whether to include reasoning in the prompt.
        allow_unsure (bool): Whether to allow the model to respond with "Unsure"
        base_model_prompt (bool): Whether to use the base model prompt that appends "Response:" to the end of the prompt.
        task_id (Optional[str]): An optional task identifier.
        template_name (str): The name of the Jinja2 template file to use for rendering.

    Returns:
        str: The generated prompt.
    """
    
    data = {
            "problem": problem,
            "answer": answer,
            "eval_correct": eval_correct,
            "include_reasoning": include_reasoning,
            "option_letters": MC_ANSWER_LETTERS,
            "allow_unsure": allow_unsure,
            "base_model_prompt": base_model_prompt,
        }
    
    if lookup_prompt_type(template_name) == PromptType.DOUBLE_CRITIC_MC:
        if not options or not answer:
            raise ValueError("For MC prompts, both 'options' and 'answer' must be provided.")
        answers, correct_index = randomize_mc_options(answer=answer, 
                                                      options=options)
        correct_answer_letter = MC_ANSWER_LETTERS[correct_index]
        data["answer"] = correct_answer_letter  # Use letter for MC answers
    else:
        answers = []
        correct_answer_letter = None
    
    data["answers"] = answers
    
    prompt_text = render_j2_template(
        data={**data, "add_instructions": True},
        template_name=template_name
    )
    problem_statement = render_j2_template(
        data={**data, "add_instructions": False},
        template_name=template_name
    )
    
    return VerificationPrompt(
        prompt_text=prompt_text,
        is_correct=is_correct,
        eval_correct=eval_correct,
        template_name=template_name,
        correct_answer=correct_answer,
        correct_answer_letter=correct_answer_letter,
        include_reasoning=include_reasoning,
        task_id=task_id,
        problem_statement=problem_statement,
        prompt_type=lookup_prompt_type(template_name),
        version=version,
        phase=phase,
        allow_unsure=allow_unsure,
        base_model_prompt=base_model_prompt,
    )


if __name__ == "__main__":

    # Example usage
    data = {
        "problem": "What is the capital of France?",
        "answer": "Paris",
        "eval_correct": True,
        "include_reasoning": False,
        "allow_unsure": False,
        "base_model_prompt": False,
    }

    print("\n\nTesting double_critic.j2 template: \n\n")
    template_name = "double_critic.j2"  # Ensure this file exists in the TEMPLATE_DIR
    
    try:
        critic_prompt = get_double_critic_prompt(
            problem=data["problem"],
            answer=data["answer"],
            is_correct=True,
            eval_correct=data["eval_correct"],
            include_reasoning=data["include_reasoning"],
            allow_unsure=data["allow_unsure"],
            base_model_prompt=data["base_model_prompt"],
        )
        print(critic_prompt.prompt_text)
    except Exception as e:
        print(f"Error: {e}")

    print("\n\nTesting generative_response.j2 template: \n\n")
    template_name = "generative_response.j2"  # Ensure this file exists in the TEMPLATE_DIR
    
    try:
        generation_prompt = get_generation_prompt(
            problem=data["problem"],
            include_reasoning=data["include_reasoning"],
            is_correct=True,
            allow_unsure=data["allow_unsure"],
            base_model_prompt=data["base_model_prompt"],
        )
        print(generation_prompt.prompt_text)
    except Exception as e:
        print(f"Error: {e}")

    print("\n\nTesting tag extraction:\n\n")
    x_data = parse_text_in_tags(keys_with_types={"answer": "str"}, 
                                text_to_search=generation_prompt.prompt_text)
    print(x_data)
   