import json
from pathlib import Path
from copy import deepcopy

class ProjectDetailsCard:
    """
    Contains project-level metadata that can be filled in separately
    and later injected into a ModelCard.
    """
    def __init__(
        self,
        model_scope_summary: str = None,
        intended_users: str = None,
        observed_limitations: str = None,
        potential_limitations: str = None,
        developed_by_name: str = None,
        developed_by_institution: str = None,
        developed_by_email: str = None,
        conflict_of_interest: str = None,
        training_data_source: str = None,
        training_data_acquisition_period: str = "",
        training_data_inclusion_exclusion_criteria: str = None,
    ):
        self.data = {
            "model_scope_summary": model_scope_summary,
            "intended_users": intended_users,
            "observed_limitations": observed_limitations,
            "potential_limitations": potential_limitations,
            "developed_by_name": developed_by_name,
            "developed_by_institution": developed_by_institution,
            "developed_by_email": developed_by_email,
            "conflict_of_interest": conflict_of_interest,
            "training_data_source": training_data_source,
            "training_data_acquisition_period": training_data_acquisition_period,
            "training_data_inclusion_exclusion_criteria": training_data_inclusion_exclusion_criteria,
        }

    def to_json(self, filename: str | Path = None) -> str:
        """Return JSON as string or save to file."""
        json_str = json.dumps(self.data, indent=2, ensure_ascii=False)
        if filename:
            Path(filename).write_text(json_str, encoding="utf-8")
        return json_str