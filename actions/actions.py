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
ALLOWED_SELECTED_VALUES = ["age", "gender", "hospital_stroke", "stroke_type",
                           "nihss_score", "thrombolysis", "no_thrombolysis_reason", "door_to_needle", "door_to_imaging",
                           "onset_to_door", "imaging_done", "imaging_type", "dysphagia_screening_type", "door_to_groin"
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
                dispatcher.utter_message(text=f"Sorry, I don't understand. Can you try rephrasing?")
                return {"plot_type": None}
            #dispatcher.utter_message(text=f"OK! I will create a {plot_type} plot.")

        PLOT_HANDLER.change_arg("type", plot_type)

        PLOT_HANDLER.send_args()

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
                dispatcher.utter_message(text=f"Sorry, I don't understand. Can you try rephrasing?")
                return {"selected_value": None}
            dispatcher.utter_message(text=f"OK! I will create a {selected_value} plot.")

        PLOT_HANDLER.change_arg("variable", selected_value)

        PLOT_HANDLER.edit_data(tracker.get_slot("nat_value"))

        return []


class ActionToggleNationalValue(Action):

    def name(self) -> Text:
        return "action_toggle_national_value"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        current_value = tracker.get_slot("nat_value")
        new_value = not current_value

        PLOT_HANDLER.change_arg("show_nat_val", new_value)
        PLOT_HANDLER.send_args()
        PLOT_HANDLER.edit_data(new_value)

        dispatcher.utter_message(text="We are {} the national value.".format("showing" if new_value else "hiding"))

        return [SlotSet("nat_value", new_value)]

class PrefillSlots(Action):
    def name(self) -> Text:
        return "action_prefill_slots"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Logic to pre-fill slots
        plot_type = "line"

        nat_value = False

        selected_value = "door_to_needle"

        real_diff = False

        return [
            SlotSet("plot_type", plot_type),
            SlotSet("nat_value", nat_value),
            SlotSet("selected_value", selected_value),
            SlotSet("real_diff", real_diff)
        ]


class ActionInitialise(Action):
    def name(self) -> Text:
        return "action_initialise"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        selected_value = tracker.get_slot("selected_value")
        nat_value = tracker.get_slot("nat_value")
        plot_type = tracker.get_slot("plot_type")

        PLOT_HANDLER.change_arg("type", plot_type)
        PLOT_HANDLER.change_arg("show_nat_val", nat_value)
        PLOT_HANDLER.change_arg("variable", selected_value)
        PLOT_HANDLER.send_args()
        PLOT_HANDLER.edit_data(nat_value)

        return []


class ActionVariableTTest(Action):
    def name(self) -> Text:
        return "action_variable_ttest"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Access dataframe from the tracker
        #data = tracker.get_slot("data")

        # Get the variable name from the tracker
        #variable_name = tracker.get_slot("variable_name")

        # Call the function to perform t-test
        p_value, cohens_d, no_2022_q2_data = PLOT_HANDLER.compare_to_past()

        # Construct message based on results
        if no_2022_q2_data:
            message = f"There was no data available for 2022 Q2. Comparing 2022 Q1 to 2021 Q4, the p-value of the wilcoxon test is {p_value:.4f}, Cohen's d is {cohens_d:.4f}."
        else:
            message = f"Comparing 2022 Q2 to 2022 Q1, the p-value of the wilcoxon test is {p_value:.4f}, Cohen's d is {cohens_d:.4f}."

        # Utter the message
        dispatcher.utter_message(text=message)

        if p_value < 0.1:
            dispatcher.utter_message(text=f"The p-value tells us that there is a {round(100 - (p_value * 100), 2)} difference between quarters, which means the difference is likely real.")
            #dispatcher.utter_message("Shall I use my A.I. abilities to find some possible causes for this difference? Or would you rather analyse the impact of this change on patient outcomes?")
            real_diff = True
        else:
            dispatcher.utter_message(text=f"The p-value tells us that there is a {round(100 - (p_value * 100), 2)} difference between quarters, which means the difference is likely cause by random chance and variance.")
            #dispatcher.utter_message("What variable would you like to explore next?")
            real_diff = False

        return [SlotSet("real_diff", real_diff)]


class ActionFindPredictors(Action):

    def name(self) -> Text:
        return "action_find_predictors"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        error, feature_weights = PLOT_HANDLER.find_predictors(True)

        if error:
            dispatcher.utter_message(f"Error occurred: {error}")
        else:
            # Round mean error and feature importances
            mean_error = round(feature_weights['Root Mean Squared Error'], 2)
            rounded_feature_weights = {feat: round(weight, 2) for feat, weight in
                                       feature_weights['Shap Values'].items()}

            # Format the feature weights as a response
            ##dispatcher.utter_message(f"Root Mean Squared Error: {mean_error}")
            ##dispatcher.utter_message("\n\nFeature Importances:\n")

            selected_value = tracker.get_slot("selected_value")
            dispatcher.utter_message(f"I built a seperate model to predict {selected_value} based only on data from your hospital")
            dispatcher.utter_message(f"The most important factors are:")
            # Send feature importances as separate messages
            features_str = ", ".join(rounded_feature_weights.keys())

            # Send a single message with all features separated by commas
            dispatcher.utter_message(features_str)

            #dispatcher.utter_message(f"Would you rather: explore how {selected_value} effects you patients or do you want to explore one of the variables I have listed?")

        return []


class ActionFindPredictorsGlobal(Action):

    def name(self) -> Text:
        return "action_global_predictions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        error, feature_weights = PLOT_HANDLER.find_predictors(False)

        if error:
            dispatcher.utter_message(f"Error occurred: {error}")
        else:
            # Round mean error and feature importances
            mean_error = round(feature_weights['Root Mean Squared Error'], 2)
            rounded_feature_weights = {feat: round(weight, 2) for feat, weight in
                                       feature_weights['Shap Values'].items()}

            # Format the feature weights as a response
            #dispatcher.utter_message(f"Root Mean Squared Error: {mean_error}")
            #dispatcher.utter_message("\n\nFeature Importances:\n")
            selected_value = tracker.get_slot("selected_value")
            dispatcher.utter_message(f"I used the data I have from multiple hospitals including your to buld a regression model that predicts {selected_value}.")
            dispatcher.utter_message(f"Based on my model, here are the indicators which influence {selected_value} the most, in order or largest to smallest influence:")
            # Send feature importances as separate messages
            # Concatenate features into a single string with commas
            features_str = ", ".join(rounded_feature_weights.keys())

            # Send a single message with all features separated by commas
            dispatcher.utter_message(features_str)

        return []


class ActionExploreEffects(Action):

    def name(self) -> Text:
        return "action_explore_effects"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        selected_value = 'discharge_mrs'
        PLOT_HANDLER.change_arg('variable', selected_value)
        PLOT_HANDLER.edit_data(False)
        PLOT_HANDLER.send_args()

        return [SlotSet("selected_value", selected_value)]


class ActionDNTGuideline(Action):
    def name(self) -> Text:
        return "action_dnt_guideline"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = PLOT_HANDLER.DNT_guideline()
        dispatcher.utter_message("The first guideline is that 75% of patients must be treated within 45 minutes..")
        dispatcher.utter_message("It looks like your hospital does meet the guideline for DNT. Should I continue checking more guidelines?")
        dispatcher.utter_message("The next one is about screening at least 90% of patients for dysphagia.")

        return []

class ActionDysphagiaGuideline(Action):
    def name(self) -> Text:
        return "action_dysphagia_guideline"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = PLOT_HANDLER.dysphagia_guideline()

        dispatcher.utter_message("While your hospital does screen many patients, you still do not meet the guideline on average. However, you did meet it for the most current quarter.")
        dispatcher.utter_message("Should we check the last guideline? It's about 90% of elligible patients getting anticoagulents.")


        return []

class ActionAnticoagsGuideline(Action):
    def name(self) -> Text:
        return "action_anticoag_guideline"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        response = PLOT_HANDLER.guideline_anticoags()

        dispatcher.utter_message("Again, your hospital provides the correct medication to many patients but does not meet the guideline of 90% except for the most redent quarter..")
        dispatcher.utter_message("This is the last of the guidelines, let me know if you would like to explore somehting else.")


        return []