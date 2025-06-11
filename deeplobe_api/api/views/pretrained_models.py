import uuid
import requests

from decouple import config

from datetime import datetime

from django.utils import timezone

from datetime import datetime, timedelta

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.pagination import PageNumberPagination

from django.db.models import Count, Max, Avg, Q
from django.db.models.functions import TruncMonth, TruncDate

from deeplobe_api.db.models import APILogger, APIKey, Workspace, AIModel,APICount,User
from deeplobe_api.api.serializers import APILoggerSerializer
from deeplobe_api.api.views.function_calls.chargebee import chargebeeplandetails


class PretrainedAPI(APIView):
    def post(self, request, kind):
        pretrained_host = config("PRETRAINED_HOST")
        mapper = {
            "pre_pose_detection": f"{pretrained_host}/pose-detection",
            "pre_pii_extraction": f"{pretrained_host}/pii-extraction",
            "pre_text_moderation": f"{pretrained_host}/text-moderation",
            "pre_facial_detection": f"{pretrained_host}/face-detection",
            "pre_table_extraction": f"{pretrained_host}/table-extractor",
            "pre_image_similarity": f"{pretrained_host}/image-similarity",
            "pre_wound_detection": f"{pretrained_host}/wound-segmentation",
            "pre_facial_expression": f"{pretrained_host}/emotion-recognition",
            "pre_sentimental_analysis": f"{pretrained_host}/sentiment-analysis",
            "pre_demographic_recognition": f"{pretrained_host}/demographic-recognition",
            "pre_people_vehicle_detection": f"{pretrained_host}/people-and-vehicle-detection",
            "pre_background_removal": f"{pretrained_host}/remove-background",
        }
        api_key = request.META.get("HTTP_APIKEY", None)

        if api_key is not None:
            api_key_user = APIKey.objects.filter(key=api_key).first()
        else:
            api_key_user = APIKey.objects.filter(user=request.user).first()

        workspace = Workspace.objects.filter(
            id=api_key_user.user.current_workspace
        ).first()
        meta_data = chargebeeplandetails(email=workspace.user)

        if api_key_user:
            if api_key_user.pretrained_model == kind:
                if (
                    meta_data["api-requests"]
                    > APILogger.objects.filter(
                        workspace=workspace,
                        created__date__range=[
                            meta_data["from_date"],
                            meta_data["to_date"],
                        ],
                    ).count()
                    and meta_data["plan_status"] == "active"
                ):
                    if (
                        kind == "pre_text_moderation"
                        or kind == "pre_sentimental_analysis"
                    ):
                        payload = {"text": request.data.get("text")}
                        api_response = requests.post(mapper.get(kind), json=payload)
                    else:
                        import io

                        payload = [
                            (
                                "file",
                                (
                                    data.name,
                                    io.BytesIO(data.file.read()),
                                    data.content_type,
                                ),
                            )
                            for data in request.FILES.getlist("file")
                        ]
                        api_response = requests.post(mapper.get(kind), files=payload)

                    logger = APILogger.objects.create(
                        uuid=uuid.uuid4().hex,
                        api_key=api_key_user,
                        model_type=kind,
                        workspace=workspace,
                        data=request.data.get("text"),
                        file=request.data.get("file"),
                    )
                    # created = logger.created
                    # updated = logger.updated
                    # response_data = api_response.json(), {
                    #     "created": created,
                    #     "updated": updated,
                    # }

                    return Response(api_response.json(), status=status.HTTP_200_OK)
            else:
                return Response(
                    {"message": "Invalid API key for requested model"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            return Response(
                {"message": "Generate api key"}, status=status.HTTP_400_BAD_REQUEST
            )

        return Response(
            {"message": "You have reached your subscription limit"},
            status=status.HTTP_400_BAD_REQUEST,
        )


class APIStatisticsByTimePeriodCircularChart(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        models = request.data.get("models")

        if not models:
            # Return an empty response when no models are provided
            return Response({"labels": [], "data": []})

        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        time_period = request.data.get("time_period")
        today = datetime.now().date()

        time_period_ranges = {
            "LAST 7 DAYS": (today - timedelta(days=7), today),
            "LAST 30 DAYS": (today - timedelta(days=30), today),
            "LAST 3 MONTHS": (today - timedelta(days=90), today),
            "LAST 6 MONTHS": (today - timedelta(days=180), today),
            "CUSTOM": None,  # Placeholder for custom date range
        }

        try:
            if time_period == "CUSTOM":
                start_date_str = request.data.get("start_date")
                end_date_str = request.data.get("end_date")
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                time_period_ranges["CUSTOM"] = (start_date, end_date)
            else:
                date_range = time_period_ranges.get(time_period)
                if date_range is None:
                    raise ValueError("Invalid time period.")
                start_date, end_date = date_range
        except Exception as e:
            return Response(str(e))

        predictions = []

        if models:
            if "ALL" in models:
                aimodel_predictions = (
                    APILogger.objects.filter(
                        workspace=workspace,
                        created__date__range=(start_date, end_date),
                        model_id__isnull=True,
                    )
                    .values("model_type")
                    .annotate(prediction_count=Count("id"))
                    .order_by("model_type")
                )
                aimodel_labels = [
                    prediction["model_type"] for prediction in aimodel_predictions
                ]
                aimodel_data = [
                    prediction["prediction_count"] for prediction in aimodel_predictions
                ]

                pretrained_predictions = (
                    APILogger.objects.filter(
                        workspace=workspace,
                        created__date__range=(start_date, end_date),
                        model_id__isnull=False,
                    )
                    .values("model_id")
                    .annotate(prediction_count=Count("id"))
                    .order_by("model_id")
                )
                pretrained_model_ids = [
                    prediction["model_id"] for prediction in pretrained_predictions
                ]
                pretrained_labels = []
                pretrained_data = []
                for model_id in pretrained_model_ids:
                    if model_id:
                        aimodel = AIModel.objects.filter(id=model_id).first()
                        if aimodel:
                            model_name = aimodel.name
                            prediction_count = APILogger.objects.filter(
                                workspace=workspace,
                                created__date__range=(start_date, end_date),
                                model_id=model_id,
                            ).count()
                            pretrained_labels.append(model_name)
                            pretrained_data.append(prediction_count)

                labels = aimodel_labels + pretrained_labels
                data = aimodel_data + pretrained_data
            else:
                for model in models:
                    if isinstance(model, int):
                        aimodel = AIModel.objects.filter(id=model).first()
                        if aimodel:
                            model_name = aimodel.name
                            prediction_count = APILogger.objects.filter(
                                workspace=workspace,
                                created__date__range=(start_date, end_date),
                                model_id=model,
                            ).count()
                            predictions.append((model_name, prediction_count))
                    elif isinstance(model, str):
                        prediction_count = APILogger.objects.filter(
                            workspace=workspace,
                            created__date__range=(start_date, end_date),
                            model_type=model,  # Filter by model_name
                        ).count()
                        predictions.append((model, prediction_count))
                    else:
                        continue
                labels = [prediction[0] for prediction in predictions]
                data = [prediction[1] for prediction in predictions]
        else:
            # No models provided, retrieve predictions for all models
            predictions = (
                APILogger.objects.filter(
                    workspace=workspace, created__date__range=(start_date, end_date)
                )
                .values("model_type")
                .annotate(prediction_count=Count("id"))
                .order_by("model_type")
            )
            labels = [prediction["model_type"] for prediction in predictions]
            data = [prediction["prediction_count"] for prediction in predictions]

        # Return the labels and data in the response
        return Response({"labels": labels, "data": data})


class APIStatisticsByTimePeriodLineChart(APIView):
    def get_time_period_range(self, time_period):
        today = timezone.now().date()
        time_period_ranges = {
            "LAST 7 DAYS": (today - timezone.timedelta(days=7), today),
            "LAST 30 DAYS": (today - timezone.timedelta(days=30), today),
            "LAST 3 MONTHS": (today - timezone.timedelta(days=90), today),
            "LAST 6 MONTHS": (today - timezone.timedelta(days=180), today),
        }
        return time_period_ranges.get(time_period)

    def post(self, request):
        pre_trained_models = {
            "pre_pose_detection" "pre_pii_extraction",
            "pre_text_moderation",
            "pre_facial_detection",
            "pre_table_extraction",
            "pre_image_similarity",
            "pre_wound_detection",
            "pre_facial_expression",
            "pre_sentimental_analysis",
            "pre_demographic_recognition",
            "pre_people_vehicle_detection",
        }

        all_months = [datetime(1900, month, 1).strftime("%B") for month in range(1, 13)]
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        time_period = request.data.get("time_period")
        time_interval = request.data.get("time_interval")
        models = request.data.get("models")
        
        if not models:
            # Return an empty response when no models are provided
            return Response({"dates": [], "datasets": []})
        if not time_period or not time_interval or not models:
            return Response({"error": "Missing required parameters."},status=status.HTTP_400_BAD_REQUEST)

        start_date = None
        end_date = None

        if time_period == "CUSTOM":
            start_date = request.data.get("start_date")
            end_date = request.data.get("end_date")
            if not start_date or not end_date:
                return Response(
                    {"error": "Missing start_date or end_date for custom time period."}
                )
            start_date = timezone.datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = timezone.datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            date_range = self.get_time_period_range(time_period)
            if date_range is None:
                return Response({"error": "Invalid time period."})
            start_date, end_date = date_range
        labels = []
        if time_interval == "DAILY":
            model_types = request.data.get("models")
            if "ALL" in model_types:
                all_model_types = (
                    APILogger.objects.filter(workspace=workspace)
                    .values_list("model_type", "model_id")
                    .distinct()
                )
                model_types = []
                for a in all_model_types:
                    if a[1] is None:
                        model_types.append(a[0])
                    else:
                        model_types.append(a[1])
                models = model_types

            else:
                model_types = set(model_types)

            data_dict = {}
            for model_type in model_types:
                data_dict[model_type] = [0] * ((end_date - start_date).days + 1)

            # Populate labels with dates
            current_date = start_date
            while current_date <= end_date:
                labels.append(current_date.strftime("%Y-%m-%d"))
                current_date += timedelta(days=1)

            # Process data for AI models
            for model_id in model_types:
                if isinstance(model_id, int):
                    # Get the AIModel object
                    try:
                        model = AIModel.objects.get(id=model_id)

                    except AIModel.DoesNotExist:
                        continue

                    model_type = model.model_type  # Get the model type
                    predictions = (
                        APILogger.objects.filter(
                            workspace=workspace,
                            created__date__range=(start_date, end_date),
                            model_id=model_id,
                        )
                        .annotate(date=TruncDate("created"))
                        .values("date")
                        .annotate(prediction_sum=Count("id"))
                        .order_by("date")
                    )

                    for prediction in predictions:
                        date = prediction["date"].strftime("%Y-%m-%d")
                        prediction_sum = prediction["prediction_sum"]
                        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                        data_dict[model_id][
                            (date_obj - start_date).days
                        ] += prediction_sum

            # Process data for pretrained models
            pretrained_models = set(model_types) - set(
                model.id for model in AIModel.objects.all()
            )
            for model_id in pretrained_models:
                predictions = (
                    APILogger.objects.filter(
                        workspace=workspace,
                        created__date__range=(start_date, end_date),
                        model_type=model_id,
                    )
                    .annotate(date=TruncDate("created"))
                    .values("date", "model_type")
                    .annotate(prediction_sum=Count("id"))
                    .order_by("date", "model_type")
                )

                for prediction in predictions:
                    date = prediction["date"].strftime("%Y-%m-%d")
                    model_type = prediction["model_type"]
                    prediction_sum = prediction["prediction_sum"]
                    date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                    data_dict[model_type][(date_obj - start_date).days] = prediction_sum

            datasets = []
            for model_type, data in data_dict.items():
                if (
                    model_type in models
                ):  # Check if the model name is in the input models
                    if (
                        model_type != pre_trained_models
                    ):  # Exclude the model from including the name field
                        if isinstance(model_type, int):
                            # Get the AIModel object
                            try:
                                model = AIModel.objects.get(id=model_type)
                            except AIModel.DoesNotExist:
                                continue
                            dataset = {
                                "label": model_type,
                                "name": model.name,
                                "type": model.model_type,
                                "data": data,
                            }
                        else:
                            dataset = {
                                "label": model_type,
                                "data": data,
                            }
                    else:
                        dataset = {
                            "label": model_type,
                            "data": data,
                        }
                    datasets.append(dataset)
                else:
                    dataset = {
                        "label": model_type,
                        "data": data,
                    }
                    datasets.append(dataset)
                    print(dataset, "set")

            response_data = {"dates": labels, "datasets": datasets}
            return Response(response_data)

        elif time_interval == "MONTHLY":
            model_types = request.data.get("models")
            if "ALL" in model_types:
                all_model_types = (
                    APILogger.objects.filter(workspace=workspace)
                    .values_list("model_type", "model_id")
                    .distinct()
                )
                model_types = []
                for a in all_model_types:
                    if a[1] is None:
                        model_types.append(a[0])
                    else:
                        model_types.append(a[1])
                models = model_types

            else:
                model_types = set(model_types)

            data_dict = {}
            for model_type in model_types:
                data_dict[model_type] = [0] * 12

            # Process data for AI models
            for model_type in model_types:
                if isinstance(model_type, int):
                    # Get the AIModel object
                    try:
                        model = AIModel.objects.get(id=model_type)
                    except AIModel.DoesNotExist:
                        continue
                    predictions = (
                        APILogger.objects.filter(
                            workspace=workspace,
                            created__date__range=(start_date, end_date),
                            model_id=model.id,
                        )
                        .annotate(month=TruncMonth("created"))
                        .values("month")
                        .annotate(prediction_sum=Count("id"))
                        .order_by("month")
                    )

                    for prediction in predictions:
                        month = prediction["month"].month
                        prediction_sum = prediction["prediction_sum"]
                        data_dict[model.id][month - 1] = prediction_sum

            # Process data for pretrained models
            pretrained_models = set(model_types) - set(
                model.id for model in AIModel.objects.all()
            )
            for model_type in pretrained_models:
                predictions = (
                    APILogger.objects.filter(
                        workspace=workspace,
                        created__date__range=(start_date, end_date),
                        model_type=model_type,
                    )
                    .annotate(month=TruncMonth("created"))
                    .values("month", "model_type")
                    .annotate(prediction_sum=Count("id"))
                    .order_by("month", "model_type")
                )

                for prediction in predictions:
                    month = prediction["month"].month
                    model_type = prediction["model_type"]
                    prediction_sum = prediction["prediction_sum"]
                    data_dict[model_type][month - 1] = prediction_sum

            datasets = []
            for model_type, data in data_dict.items():
                if (
                    model_type in models
                ):  # Check if the model name is in the input models
                    if (
                        model_type != pre_trained_models
                    ):  # Exclude the model from including the name field
                        if isinstance(model_type, int):
                            # Get the AIModel object
                            try:
                                model = AIModel.objects.get(id=model_type)
                            except AIModel.DoesNotExist:
                                continue
                            dataset = {
                                "label": model_type,
                                "name": model.name,
                                "type": model.model_type,
                                "data": data,
                            }
                        else:
                            dataset = {
                                "label": model_type,
                                "data": data,
                            }
                    else:
                        dataset = {
                            "label": model_type,
                            "data": data,
                        }
                    datasets.append(dataset)
                else:
                    dataset = {
                        "label": model_type,
                        "data": data,
                    }
                    datasets.append(dataset)

            response_data = {"months": all_months, "datasets": datasets}
            return Response(response_data)


class APIStatistics(APIView):
    permission_classes = [IsAuthenticated]

    def get_most_predicted_model(self, request):
        user_obj = User.objects.filter(email=request.user.email).first()
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        apicount = APICount.objects.filter(workspace=workspace.id)

        if apicount.exists():
            apicalls = apicount[0].custom_model_api_count + APILogger.objects.filter(api_key__user=user_obj).count()
        else:
            apicalls = APILogger.objects.filter(api_key__user=user_obj).count()

        # Get count of active AI models with status "Live"
        active_model_count = AIModel.objects.filter(
            workspace=request.user.current_workspace, status="Live"
        ).count()

        most_predicted_model = (
            APILogger.objects.values("model_type").filter(workspace=workspace)
            .annotate(prediction_count=Count("model_type"))
            .order_by("-prediction_count")
            .first()
        )

        if most_predicted_model is not None:
            model_name = most_predicted_model["model_type"]
            model_count =  apicalls

            # Calculate the total count of active AI models with "Live" status
            active_models = active_model_count

            return {
                "most_predicted_model": model_name,
                "prediction_count": model_count,
                "active_models": active_models,
            }
        else:
            return {
                "most_predicted_model": None,
                "prediction_count": 0,
                "active_models": 0,
            }

    def get_monthly_averages(self, workspace):
        total_count = APILogger.objects.filter(workspace=workspace).count()
        monthly_averages = (
            APILogger.objects.annotate(month=TruncMonth("created"))
            .filter(workspace=workspace)
            .values("month")
            .annotate(
                prediction_count=Count("model_type"),
                prediction_average=Avg(total_count),
            )
            .order_by("month")
        )
        results = []
        for monthly_average in monthly_averages:
            month = monthly_average["month"]
            prediction_count = monthly_average["prediction_count"]
            prediction_average = monthly_average["prediction_average"]
            result = {
                "Month": month,
                "Count": prediction_count,
                "Average": prediction_average,
            }
            results.append(result)
        return results

    def get(self, request):
        most_predicted_model = self.get_most_predicted_model(request)
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()
        monthly_averages = self.get_monthly_averages(workspace)
        active_models = most_predicted_model["active_models"]
        return Response(
            {
                "most_predicted_model": most_predicted_model["most_predicted_model"],
                "prediction_count": most_predicted_model["prediction_count"],
                "monthly_averages": monthly_averages,
                "active_models": active_models,
            }
        )


class APILoggerView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of all APILogger count.
        """
        try:
            workspace = Workspace.objects.filter(
                id=request.user.current_workspace
            ).first()
            qs = (
                APILogger.objects.filter(workspace=workspace)
                .values("model_type")
                .annotate(model_type_count=Count("model_type"), created=Max("created"))
            )
            total_count = APILogger.objects.filter(workspace=workspace).count()
            return Response(
                {"list": list(qs), "pretrained_total_count": total_count},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            print(e)
            return Response([], status=status.HTTP_400_BAD_REQUEST)
        
        
class APIKeyModelsView(APIView):
    permission_classes = [IsAuthenticated]
      
    def get(self, request):
        workspace = Workspace.objects.filter(id=request.user.current_workspace).first()

        # Get the count of model_type for custom models and pretrained models
        custom_counts = APILogger.objects.filter(
            Q(workspace=workspace) & Q(is_custom_model=True)
        ).values("model_id").annotate(model_type_count=Count("model_id"))

        pretrained_counts = APILogger.objects.filter(
            Q(workspace=workspace) & Q(is_custom_model=False) & Q(model_id__isnull=True)
        ).values("model_type").annotate(model_type_count=Count("model_type"))

        # Get the name and type for custom models
        custom_models = AIModel.objects.filter(id__in=[item["model_id"] for item in custom_counts])
        custom_model_mapping = {model.id: model for model in custom_models}

        # Combine the results for custom models and pretrained models
        results = []
        for item in custom_counts:
            model_id = item["model_id"]
            # model_type_count = item["model_type_count"]
            if model_id in custom_model_mapping:
                custom_model = custom_model_mapping[model_id]
                results.append({
                    "model_id": model_id,
                    "model_name": custom_model.name,
                    "model_type": custom_model.get_model_type_display(),
                    # "model_type_count": model_type_count,
                })

        for item in pretrained_counts:
            results.append({
                "model_type": item["model_type"],
                # "model_type_count": item["model_type_count"],
            })

        return Response({"list": results}, status=status.HTTP_200_OK)
    
class CustomPagination(PageNumberPagination):
    page_size = 10  # Adjust the page size according to your needs
    page_size_query_param = "page_size"
    max_page_size = 100

    def get_page_number(self, request, paginator):
        """
        Override the get_page_number method to retrieve the page number
        from the request data.
        """
        page_number = request.data.get("page", 1)
        return page_number


class APIMetricsLogsView(APIView):
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination()

    def post(self, request):
        try:
            workspace = Workspace.objects.filter(
                id=request.user.current_workspace
            ).first()
            qs = (
                APILogger.objects.filter(workspace=workspace)
                .values("model_id", "model_type", "response_code", "created")
                .order_by("-created")
            )

            # Get query params for filtering
            time_period = request.data.get("time_period", "All")
            start_date = None
            end_date = None
            model_type = request.data.get("models", None)
            if not model_type:
                # Return an empty response when no models are provided
                return Response({"count": [], "results": []})

            # Filter logs based on time range
            if time_period == "LAST 7 DAYS":
                last_7_days = datetime.now().date() - timedelta(days=7)
                qs = qs.filter(created__date__gte=last_7_days)
            elif time_period == "LAST 30 DAYS":
                last_30_days = datetime.now().date() - timedelta(days=30)
                qs = qs.filter(created__date__gte=last_30_days)
            elif time_period == "LAST 3 MONTHS":
                last_3_months = datetime.now().date() - timedelta(days=90)
                qs = qs.filter(created__date__gte=last_3_months)
            elif time_period == "LAST 6 MONTHS":
                last_6_months = datetime.now().date() - timedelta(days=180)
                qs = qs.filter(created__date__gte=last_6_months)
            elif time_period == "CUSTOM":
                start_date = request.data.get("start_date")
                end_date = request.data.get("end_date")
                if start_date and end_date:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                    end_date = datetime.strptime(
                        end_date, "%Y-%m-%d"
                    ).date() + timedelta(days=1)
                    qs = qs.filter(created__date__range=[start_date, end_date])

            if model_type and "ALL" not in map(str.upper, map(str, model_type)):
                model_types = set(model_type)
                ai_model_ids = set()
                pretrained_models = set()
                for model_type in model_types:
                    try:
                        model_id = int(model_type)
                        ai_model_ids.add(model_id)
                    except ValueError:
                        pretrained_models.add(model_type)

                # Filter logs based on AI model IDs
                ai_model_logs = qs.filter(Q(model_id__in=ai_model_ids))

                # Filter logs based on pretrained model strings
                pretrained_model_logs = qs.filter(
                    Q(model_type__in=pretrained_models)
                    | Q(model_id__model_type__in=pretrained_models)
                )

                # Combine the querysets
                combined_logs = ai_model_logs | pretrained_model_logs

                # Paginate the combined queryset
                paginated_logs = self.pagination_class.paginate_queryset(
                    combined_logs, request, view=self
                )

                # Serialize the log data
                log_data = [
                    {
                        "model_type": AIModel.objects.get(id=log["model_id"]).name
                        if log["model_id"]
                        else log["model_type"],
                        "response_code": 200
                        if log["response_code"] is None
                        else log["response_code"],
                        "created": log["created"],
                    }
                    for log in paginated_logs
                ]

                return self.pagination_class.get_paginated_response(log_data)

            # If "models" input contains "ALL", retrieve logs for all models without filtering by model type
            # Paginate the queryset
            paginated_qs = self.pagination_class.paginate_queryset(
                qs, request, view=self
            )

            # Serialize the log data
            log_data = [
                {
                    "model_type": AIModel.objects.get(id=log["model_id"]).name
                    if log["model_id"]
                    else log["model_type"],
                    "response_code": 200
                    if log["response_code"] is None
                    else log["response_code"],
                    "created": log["created"],
                }
                for log in paginated_qs
            ]

            return self.pagination_class.get_paginated_response(log_data)

        except Exception as e:
            print(e)
            return Response([], status=status.HTTP_400_BAD_REQUEST)


class APILoggerViewAll(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, type):
        """
        Return a list of all APILogger.
        """
        qs = APILogger.objects.filter(
            api_key__user=request.user,
            model_type=type,
            workspace=request.user.current_workspace,
        )
        serializer = APILoggerSerializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class APILoggerCustomDetails(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, model_id):
        """
        Return a list of all APILogger.
        """
        qs = APILogger.objects.filter(
            api_key__user=request.user,
            model_id=model_id,
            workspace=request.user.current_workspace,
        )
        serializer = APILoggerSerializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
