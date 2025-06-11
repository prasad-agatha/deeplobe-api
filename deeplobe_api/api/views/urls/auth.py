from django.urls import path

from deeplobe_api.api.views.auth import (
    IsAdmin,
    Sendmagiclink,
    SocialAuthView,
    VerifyMagicLink,
    ModelPermissions,
    MagicLinkRegister,
    UserDeleteByAdmin,
    ActivateUserEndpoint,
    # UpdateFirstTimeUser
)


urlpatterns = [
    # auth endpoints
    path("gmail-auth/", SocialAuthView.as_view(), name="social_authentication"),
    path("magiclink/", Sendmagiclink.as_view()),
    path("verifymagiclink/", VerifyMagicLink.as_view()),
    path("magiclinkregister/", MagicLinkRegister.as_view()),
    path(
        "users/<int:id>/activate/",
        ActivateUserEndpoint.as_view(),
        name="active-user",
    ),
    path(
        "modelpermissions/",
        ModelPermissions.as_view(),
        name="model_permissions",
    ),
    path(
        "admin/",
        IsAdmin.as_view(),
        name="is_admin",
    ),
    path(
        "userdeletedbyadmin/<int:pk>",
        UserDeleteByAdmin.as_view(),
        name="is_admin",
    ),
    # path("updatefirsttimeuser/", UpdateFirstTimeUser.as_view(), name="first-time-user"),
]
