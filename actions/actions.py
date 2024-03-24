# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
from rasa.core.actions.forms import FormAction

# This is a simple example for a custom action which utters "Hello World!"

from actions.utils import plot_handler
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.types import DomainDict

# SOCKET = sockets.Socket()
PLOT_HANDLER = plot_handler.PlotHandler()
ALLOWED_PLOT_TYPES = ["line", "bar", "pie", "barh"]
ALLOWED_SELECTED_VALUES = ["age", "gender", "hospital_stroke", "hospitalized_in", "department_type", "stroke_type",
                           "nihss_score", "thrombolysis", "no_thrombolysis_reason", "door_to_needle", "door_to_imaging",
                           "onset_to_door", "imaging_done", "imaging_type", "dysphagia_screening_type",
                           "before_onset_antidiabetics", "before_onset_cilostazol", "before_onset_clopidrogel",
                           "before_onset_ticagrelor", "before_onset_ticlopidine", "before_onset_prasugrel",
                           "before_onset_dipyridamol", "before_onset_warfarin", "risk_hypertension", "risk_diabetes",
                           "risk_hyperlipidemia", "risk_congestive_heart_failure", "risk_smoker",
                           "risk_previous_ischemic_stroke", "risk_previous_hemorrhagic_stroke",
                           "risk_coronary_artery_disease_or_myocardial_infarction", "risk_hiv", "bleeding_source",
                           "discharge_mrs", "discharge_nihss_score", "three_m_mrs", "covid_test",
                           "physiotherapy_start_within_3days", "occup_physiotherapy_received", "glucose", "cholesterol",
                           "sys_blood_pressure", "dis_blood_pressure", "perfusion_core", "hypoperfusion_core",
                           "stroke_mimics_diagnosis", "prestroke_mrs", "tici_score", "prenotification", "ich_score",
                           "hunt_hess_score"]



class ActionChangePlottype(Action):

    def name(self) -> Text:
        return "action_change_plottype"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        plot_type = tracker.get_slot("plot_type")

        print(plot_type)

        if plot_type:
            if plot_type.lower() not in ALLOWED_PLOT_TYPES:
                dispatcher.utter_message(text=f"Sorry, I can only create {'/'.join(ALLOWED_PLOT_TYPES)} plots.")
                return {"plot_type": None}
            dispatcher.utter_message(text=f"OK! I will create a {plot_type} plot.")

        PLOT_HANDLER.change_arg("type", plot_type)

        response = PLOT_HANDLER.send_args()
        dispatcher.utter_message(text=f"{response}")

        return []


class ActionChangeSelectedvalue(Action):

    def name(self) -> Text:
        return "action_change_selectedvalue"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        selected_value = tracker.get_slot("selected_value")

        if selected_value:
            if selected_value.lower() not in ALLOWED_SELECTED_VALUES:
                dispatcher.utter_message(text=f"Sorry, I can only create {'/'.join(ALLOWED_SELECTED_VALUES)} plots.")
                return {"selected_value": None}
            dispatcher.utter_message(text=f"OK! I will create a {selected_value} plot.")

        PLOT_HANDLER.change_arg("variable", selected_value)

        response = PLOT_HANDLER.edit_data()
        dispatcher.utter_message(text=f"{response}")

        return []


class PrefillSlots(Action):
    def name(self) -> Text:
        return "action_prefill_slots"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Logic to pre-fill slots
        plot_type = "line"

        return [
            SlotSet("plot_type", plot_type)
        ]


class ActionVariableTTest(Action):
    def name(self) -> Text:
        return "action_variable_ttest"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Access dataframe from the tracker
        data = tracker.get_slot("data")

        # Get the variable name from the tracker
        variable_name = tracker.get_slot("variable_name")

        # Call the function to perform t-test
        p_value, cohens_d, no_2022_q2_data = PLOT_HANDLER.compare_to_past()

        # Construct message based on results
        if no_2022_q2_data:
            message = f"There was no data available for 2022 Q2. Comparing 2022 Q1 to 2021 Q2, the p-value of the t-test is {p_value:.4f}, Cohen's d is {cohens_d:.4f}."
        else:
            message = f"Comparing 2022 Q2 to 2022 Q1, the p-value of the t-test is {p_value:.4f}, Cohen's d is {cohens_d:.4f}."

        # Utter the message
        dispatcher.utter_message(text=message)

        return []


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "ActionHelloWorld"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Here is your INFO")

        return []

